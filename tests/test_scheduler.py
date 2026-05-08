"""Scheduler unit tests."""

from hpc.scheduler import PJM, Slurm, JobStatus


class TestSlurmStatusCmd:
    def test_status_cmd_includes_X_flag(self):
        scheduler = Slurm()

        cmd = scheduler.status_cmd("12345678")

        # -X suppresses jobsteps so each row is one allocation; without it
        # array-task aggregation would conflate steps with tasks.
        assert "-X" in cmd
        assert "sacct" in cmd
        assert "12345678" in cmd
        assert "--noheader" in cmd


class TestSlurmParseStatus:
    # Single-line cases — preserves backward compat with single-job mocks
    # and exercises every entry in the status map.

    def test_single_completed(self):
        assert Slurm().parse_status("COMPLETED\n") == JobStatus.COMPLETED

    def test_single_pending(self):
        assert Slurm().parse_status("PENDING\n") == JobStatus.PENDING

    def test_single_running(self):
        assert Slurm().parse_status("RUNNING\n") == JobStatus.RUNNING

    def test_single_failed(self):
        assert Slurm().parse_status("FAILED\n") == JobStatus.FAILED

    def test_single_cancelled(self):
        assert Slurm().parse_status("CANCELLED\n") == JobStatus.CANCELLED

    def test_single_timeout(self):
        assert Slurm().parse_status("TIMEOUT\n") == JobStatus.TIMEOUT

    def test_single_requeued_is_non_terminal(self):
        # REQUEUED is a transient state where Slurm has put the job back in
        # the queue; wait_for_job must keep polling, not exit.
        assert Slurm().parse_status("REQUEUED\n") == JobStatus.PENDING

    def test_single_suspended_is_non_terminal(self):
        assert Slurm().parse_status("SUSPENDED\n") == JobStatus.RUNNING

    def test_single_completing_is_non_terminal(self):
        # COMPLETING means the job is finishing cleanup but processes may
        # still be active on some nodes; treating it as terminal would
        # exit wait_for_job before output files are flushed.
        assert Slurm().parse_status("COMPLETING\n") == JobStatus.RUNNING

    def test_single_configuring_is_non_terminal(self):
        assert Slurm().parse_status("CONFIGURING\n") == JobStatus.PENDING

    def test_single_resizing_is_non_terminal(self):
        assert Slurm().parse_status("RESIZING\n") == JobStatus.RUNNING

    def test_single_boot_fail(self):
        assert Slurm().parse_status("BOOT_FAIL\n") == JobStatus.FAILED

    def test_single_node_fail(self):
        assert Slurm().parse_status("NODE_FAIL\n") == JobStatus.FAILED

    def test_single_out_of_memory(self):
        assert Slurm().parse_status("OUT_OF_MEMORY\n") == JobStatus.FAILED

    def test_single_deadline(self):
        assert Slurm().parse_status("DEADLINE\n") == JobStatus.TIMEOUT

    def test_single_preempted(self):
        assert Slurm().parse_status("PREEMPTED\n") == JobStatus.CANCELLED

    def test_single_revoked(self):
        assert Slurm().parse_status("REVOKED\n") == JobStatus.CANCELLED

    # Trailing-+ handling — sacct appends a bare "+" to some states
    # (e.g. "COMPLETED+", "CANCELLED+") to flag extra context;
    # rstrip("+") must apply to every aggregated row, not just the first.

    def test_completed_with_plus_suffix(self):
        assert Slurm().parse_status("COMPLETED+\n") == JobStatus.COMPLETED

    def test_aggregation_strips_plus_on_every_row(self):
        output = "COMPLETED+\nCOMPLETED+\nCOMPLETED+\n"
        assert Slurm().parse_status(output) == JobStatus.COMPLETED

    # Empty / unknown — empty output preserves the pre-existing FAILED
    # behavior; unknown states fall back to FAILED so we stay
    # conservative with respect to wait termination.

    def test_empty_output(self):
        assert Slurm().parse_status("") == JobStatus.FAILED

    def test_whitespace_only_output(self):
        assert Slurm().parse_status("   \n  \n") == JobStatus.FAILED

    def test_unknown_status_falls_back_to_failed(self):
        assert Slurm().parse_status("BOGUS_STATE\n") == JobStatus.FAILED

    # Aggregation — array-job scenarios. The invariant under test is
    # aggregate(S) ∈ terminal_states ⟺ ∀ s ∈ S, s ∈ terminal_states.

    def test_aggregate_all_completed(self):
        output = "COMPLETED\nCOMPLETED\nCOMPLETED\n"
        assert Slurm().parse_status(output) == JobStatus.COMPLETED

    def test_aggregate_one_pending_others_completed(self):
        # Regression guard: the previous parse_status only inspected lines[0]
        # and returned COMPLETED while later tasks were still pending.
        output = "COMPLETED\nCOMPLETED\nPENDING\n"
        assert Slurm().parse_status(output) == JobStatus.PENDING

    def test_aggregate_pending_first_others_running(self):
        # Even when PENDING appears first, RUNNING wins (RUNNING > PENDING).
        output = "PENDING\nRUNNING\nCOMPLETED\n"
        assert Slurm().parse_status(output) == JobStatus.RUNNING

    def test_aggregate_one_running_dominates_pending(self):
        output = "PENDING\nPENDING\nRUNNING\nCOMPLETED\n"
        assert Slurm().parse_status(output) == JobStatus.RUNNING

    def test_aggregate_requeued_keeps_array_non_terminal(self):
        # A single REQUEUED task in an otherwise-completed array must not
        # mark the array terminal; wait_for_job must keep polling.
        output = "COMPLETED\nCOMPLETED\nREQUEUED\n"
        assert Slurm().parse_status(output) == JobStatus.PENDING

    def test_aggregate_terminal_failure_dominates_completed(self):
        # All tasks terminal but at least one FAILED → array is FAILED.
        output = "COMPLETED\nCOMPLETED\nFAILED\n"
        assert Slurm().parse_status(output) == JobStatus.FAILED

    def test_aggregate_failed_dominates_cancelled(self):
        output = "COMPLETED\nCANCELLED\nFAILED\n"
        assert Slurm().parse_status(output) == JobStatus.FAILED

    def test_aggregate_cancelled_dominates_timeout(self):
        # FAILED > CANCELLED > TIMEOUT — when no FAILED present,
        # CANCELLED wins over TIMEOUT.
        output = "COMPLETED\nTIMEOUT\nCANCELLED\n"
        assert Slurm().parse_status(output) == JobStatus.CANCELLED

    def test_aggregate_running_beats_failed_terminal(self):
        # Even with a failed task, an in-flight task must keep the
        # aggregate non-terminal so wait_for_job does not exit while
        # other tasks are still running.
        output = "FAILED\nRUNNING\nCOMPLETED\n"
        assert Slurm().parse_status(output) == JobStatus.RUNNING

    def test_aggregate_pending_beats_failed_terminal(self):
        output = "FAILED\nPENDING\nCOMPLETED\n"
        assert Slurm().parse_status(output) == JobStatus.PENDING


class TestPJMParseJobID:
    def test_parse_job_id_prefers_job_token(self):
        scheduler = PJM()

        output = "[INFO] PJM 0000 pjsub Job 12345678 submitted."

        assert scheduler.parse_job_id(output) == "12345678"

    def test_parse_job_id_falls_back_to_last_numeric_token(self):
        scheduler = PJM()

        output = "PJM 0000: submitted job 87654321"

        assert scheduler.parse_job_id(output) == "87654321"

    def test_parse_job_id_returns_stripped_output_when_no_numeric_token(self):
        scheduler = PJM()

        output = " unexpected output "

        assert scheduler.parse_job_id(output) == "unexpected output"
