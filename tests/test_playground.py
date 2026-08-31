"""The job lifecycle: a wedged pipeline must not wedge the server forever.

`_run_job` runs the whole multi-minute pipeline inside a daemon thread while
holding a process-global `_LOCK` -- by design, only one pipeline runs at a
time. Before this file existed, a raster read that never returns (a FIFO
inside an uploaded directory, say) left that thread parked forever, `_LOCK`
never released, and every job behind it queued in silence with no error and
no way to tell what happened; only a process restart recovered. Separately,
`_JOBS` never evicted anything, so `Job.body` -- the full rendered HTML report,
~3.2 MB per run -- stayed resident for the life of the process no matter how
many runs went by.

These tests exercise both fixes without ever sleeping for the real
`JOB_TIMEOUT_S` (600 s): the timeout is a module attribute `_run_job` reads
fresh on every call, so a test can monkeypatch it down to a fraction of a
second first.
"""

import time

import pytest

from segmap_digest import playground as pg


@pytest.fixture(autouse=True)
def _isolated_jobs(monkeypatch):
    """Every test gets its own empty `_JOBS`, so job ids, eviction counts, and
    `/jobs` history from one test can never leak into another."""
    monkeypatch.setattr(pg, "_JOBS", {})


# --- the lock-acquire timeout: a job queued behind a wedged run -------------

def test_a_queued_job_fails_clearly_instead_of_waiting_forever(monkeypatch, tmp_path):
    """Simulates the wedged-pipeline scenario: something else holds `_LOCK`
    and never releases it (a hung `rasterio.open()` never returns control to
    Python at all, so `_LOCK.release()` in its `finally` never runs either).

    The acceptance bar from the review: a new submission must not queue in
    silence forever. It must give up after JOB_TIMEOUT_S and say so plainly,
    even though `_LOCK` itself stays stuck -- that part is not fixable from
    here, which is exactly why this test checks for a clear failure and NOT
    for `_LOCK` becoming free.
    """
    monkeypatch.setattr(pg, "JOB_TIMEOUT_S", 0.1)
    assert pg._LOCK.acquire(blocking=False), "test setup: lock must start free"
    try:
        job = pg._new_job("queued-behind-a-wedged-run")
        work = tmp_path / "work"
        work.mkdir()

        start = time.time()
        pg._run_job(job, tmp_path / "in.tif", {}, tmp_path / "cache", work)
        elapsed = time.time() - start

        assert elapsed < 5.0, "must give up, not hang for the test's lifetime"
        assert job.status == "failed"
        assert "gave up" in job.error.lower()
        assert "stuck" in job.error.lower()
        assert "restart" in job.error.lower()
        # the job never actually ran the pipeline
        assert job.body is None
        assert job.out_dir is None
        # honesty check: the wedged holder's lock is still not released --
        # this fix bounds the QUEUE, it does not reclaim a hung syscall
        assert pg._LOCK.locked()
    finally:
        pg._LOCK.release()


def test_a_free_lock_is_unaffected_by_the_timeout(monkeypatch, tmp_path):
    """The timeout path must only trigger when the lock is actually contended
    -- an ordinary run, with nothing else holding `_LOCK`, must not pay any
    waiting cost or be misreported as timed out."""
    monkeypatch.setattr(pg, "JOB_TIMEOUT_S", 0.1)

    def fast_pipeline(path, opts, cache_dir, job=None):
        return pg.Run(source="synthetic")

    monkeypatch.setattr(pg, "run_pipeline", fast_pipeline)
    monkeypatch.setattr(pg, "OUT_ROOT", tmp_path / "out")

    job = pg._new_job("ordinary-run")
    work = tmp_path / "work"
    work.mkdir()
    pg._run_job(job, tmp_path / "in.tif", {}, tmp_path / "cache", work)

    assert job.status == "done"
    assert job.error == ""
    assert not pg._LOCK.locked()


# --- the watchdog: a run that is itself the one stuck -----------------------

def test_watchdog_fails_a_job_still_running_past_the_deadline(monkeypatch, tmp_path):
    """This is the OTHER half of the timeout story: the job that IS running
    (not one queued behind another) outlives JOB_TIMEOUT_S. The watchdog must
    report it failed so `/job/<id>` stops saying "running" forever -- but if
    the underlying call eventually does return (as this fake one does, after
    outliving the deadline), `_LOCK` is released normally by the thread that
    held it. The two are independent: the page recovers immediately at the
    deadline, `_LOCK` recovers only once -- if ever -- the call returns.
    """
    monkeypatch.setattr(pg, "JOB_TIMEOUT_S", 0.05)

    def slow_pipeline(path, opts, cache_dir, job=None):
        time.sleep(0.4)          # outlives the 0.05s deadline
        return pg.Run(source="synthetic")

    monkeypatch.setattr(pg, "run_pipeline", slow_pipeline)
    monkeypatch.setattr(pg, "OUT_ROOT", tmp_path / "out")

    job = pg._new_job("slow-run")
    work = tmp_path / "work"
    work.mkdir()
    pg._run_job(job, tmp_path / "in.tif", {}, tmp_path / "cache", work)

    assert job.status == "failed"
    assert "timed out" in job.error.lower()
    # the call DID eventually return -- _LOCK must be released like any
    # ordinary finished run, and the report is still written to disk even
    # though the user was already told it failed
    assert not pg._LOCK.locked()
    assert job.out_dir is not None
    assert (job.out_dir / "index.html").exists()


def test_watchdog_is_a_no_op_once_the_job_already_finished(monkeypatch):
    """A watchdog firing after a job already reached a terminal state must
    never stomp its real outcome."""
    job = pg.Job(id="j-done", name="n", status="done")
    pg._mark_timed_out(job, 1.0)
    assert job.status == "done"
    assert job.error == ""

    job2 = pg.Job(id="j-failed", name="n", status="failed", error="boom")
    pg._mark_timed_out(job2, 1.0)
    assert job2.status == "failed"
    assert job2.error == "boom"


# --- the LRU cap on _JOBS ----------------------------------------------------

def test_eviction_drops_body_from_the_oldest_finished_jobs_past_the_cap(monkeypatch):
    monkeypatch.setattr(pg, "MAX_JOBS_RETAINED", 3)
    done_ids = [f"20260101-000000-{i:02d}" for i in range(5)]
    for jid in done_ids:
        pg._JOBS[jid] = pg.Job(id=jid, name=jid, status="done", body=b"x" * 100)

    running = pg.Job(id="20260101-000006-01", name="r", status="running")
    waiting = pg.Job(id="20260101-000007-01", name="w", status="waiting")
    pg._JOBS[running.id] = running
    pg._JOBS[waiting.id] = waiting

    pg._evict_old_jobs()

    # 7 jobs total, cap 3 -> 4 evicted; oldest ids go first
    bodies = [pg._JOBS[jid].body for jid in done_ids]
    assert bodies[:4] == [None, None, None, None]
    assert bodies[4] == b"x" * 100, "the newest finished job must survive"

    # a running/waiting job is never touched, even though it counted toward
    # the overflow that triggered eviction in the first place
    assert running.status == "running" and running.body is None
    assert waiting.status == "waiting" and waiting.body is None

    # the Job *records* are not deleted -- only the body -- so /jobs history
    # and /result/<id>'s out_dir lookup both still work
    assert set(pg._JOBS) == set(done_ids) | {running.id, waiting.id}


def test_eviction_is_a_no_op_under_the_cap(monkeypatch):
    monkeypatch.setattr(pg, "MAX_JOBS_RETAINED", 50)
    job = pg.Job(id="only-one", name="n", status="done", body=b"keep-me")
    pg._JOBS[job.id] = job
    pg._evict_old_jobs()
    assert job.body == b"keep-me"


def test_a_completed_jobs_report_survives_its_own_eviction(monkeypatch, tmp_path):
    """End to end: run two jobs with the cap set to 1, and confirm the older
    one's report is still servable -- via `_report_bytes`, the same call
    `/result/<id>` makes -- after eviction has dropped its in-memory body."""
    monkeypatch.setattr(pg, "MAX_JOBS_RETAINED", 1)
    monkeypatch.setattr(pg, "OUT_ROOT", tmp_path / "out")

    def fake_pipeline(path, opts, cache_dir, job=None):
        return pg.Run(source="synthetic")

    monkeypatch.setattr(pg, "run_pipeline", fake_pipeline)

    job_a = pg._new_job("run-a")
    work_a = tmp_path / "work-a"
    work_a.mkdir()
    pg._run_job(job_a, tmp_path / "a.tif", {}, tmp_path / "cache", work_a)
    assert job_a.status == "done" and job_a.body is not None
    persisted = (job_a.out_dir / "index.html").read_bytes()

    job_b = pg._new_job("run-b")
    work_b = tmp_path / "work-b"
    work_b.mkdir()
    pg._run_job(job_b, tmp_path / "b.tif", {}, tmp_path / "cache", work_b)

    assert job_a.body is None, "the older finished job's body should be evicted"
    assert pg._report_bytes(job_a) == persisted
    # the newer job is still under the cap and keeps serving straight from memory
    assert job_b.body is not None
    assert pg._report_bytes(job_b) == job_b.body


# --- _report_bytes in isolation ---------------------------------------------

def test_report_bytes_prefers_the_in_memory_body(tmp_path):
    out_dir = tmp_path / "out-dir"
    out_dir.mkdir()
    (out_dir / "index.html").write_bytes(b"stale-on-disk")
    job = pg.Job(id="j", name="n", status="done", out_dir=out_dir,
                 body=b"fresh-in-memory")
    assert pg._report_bytes(job) == b"fresh-in-memory"


def test_report_bytes_falls_back_to_the_persisted_file(tmp_path):
    out_dir = tmp_path / "out-dir"
    out_dir.mkdir()
    (out_dir / "index.html").write_bytes(b"<html>persisted report</html>")
    job = pg.Job(id="j", name="n", status="done", out_dir=out_dir, body=None)
    assert pg._report_bytes(job) == b"<html>persisted report</html>"


def test_report_bytes_is_none_when_there_is_nothing_to_serve():
    assert pg._report_bytes(pg.Job(id="j", name="n", status="running")) is None
    # out_dir set but the file is missing (e.g. wiped from under it)
    job = pg.Job(id="j2", name="n", status="done", body=None)
    assert pg._report_bytes(job) is None
