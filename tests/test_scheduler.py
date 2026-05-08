"""Scheduler unit tests."""

from hpc.scheduler import PJM, JobDetail, JobStatus, Slurm


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

    def test_status_cmd_widens_state_column(self):
        # sacct's default column width is 10 chars, which truncates long
        # state names like CONFIGURING / OUT_OF_MEMORY into "CONFIGUR+" and
        # silently maps them to FAILED via the unknown-state fallback.
        # Pin the explicit width so the format never narrows back.
        cmd = Slurm().status_cmd("12345678")

        assert "--format=State%30" in cmd


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

    def test_padded_output_from_widened_state_column(self):
        # sacct with --format=State%30 right-pads each row with spaces;
        # the per-row strip() must reduce the padded value to its bare
        # state name before the status-map lookup.
        output = "CONFIGURING                   \nOUT_OF_MEMORY                 \n"
        # CONFIGURING is non-terminal (PENDING); aggregate stays non-terminal
        # even though OUT_OF_MEMORY is terminal failure.
        assert Slurm().parse_status(output) == JobStatus.PENDING

    def test_cancelled_by_user_extended_form(self):
        # With the widened State%30 column, sacct emits the full
        # "CANCELLED by <uid>" form instead of truncating to "CANCELLED+".
        # Taking the first whitespace-separated token reduces it to
        # "CANCELLED" so the status map matches.
        output = "CANCELLED by 12345\n"
        assert Slurm().parse_status(output) == JobStatus.CANCELLED

    def test_aggregate_with_cancelled_by_user(self):
        output = "COMPLETED\nCOMPLETED\nCANCELLED by 12345\n"
        assert Slurm().parse_status(output) == JobStatus.CANCELLED


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


class TestSlurmDetailCmd:
    def test_detail_cmd_includes_required_format_and_parsable_flag(self):
        cmd = Slurm().detail_cmd("12345")
        assert cmd is not None
        assert cmd[0] == "sacct"
        assert "12345" in cmd
        assert "--format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem" in cmd
        assert "--noheader" in cmd
        assert "-P" in cmd


class TestSlurmParseDetail:
    def test_parse_typical_three_row_output(self):
        # parent (MaxRSS empty) + .batch (carries MaxRSS) + .extern.
        output = (
            "12345|COMPLETED|0:0|00:01:23||16Gn\n"
            "12345.batch|COMPLETED|0:0|00:01:23|1024K|16Gn\n"
            "12345.extern|COMPLETED|0:0|00:01:23|0|16Gn\n"
        )
        detail = Slurm().parse_detail(output)
        assert detail == JobDetail(
            state="COMPLETED",
            exit_code="0:0",
            elapsed="00:01:23",
            max_rss="1024K",
            req_mem="16Gn",
        )

    def test_parse_falls_back_to_first_nonempty_max_rss_when_no_batch_row(self):
        output = (
            "12345|COMPLETED|0:0|00:00:42||8Gn\n"
            "12345.0|COMPLETED|0:0|00:00:42|512K|8Gn\n"
        )
        detail = Slurm().parse_detail(output)
        assert detail is not None
        assert detail.max_rss == "512K"

    def test_parse_tolerates_trailing_pipe_from_parsable_mode(self):
        output = "12345|COMPLETED|0:0|00:00:10||4Gn|\n12345.batch|COMPLETED|0:0|00:00:10|256K|4Gn|\n"
        detail = Slurm().parse_detail(output)
        assert detail == JobDetail(
            state="COMPLETED",
            exit_code="0:0",
            elapsed="00:00:10",
            max_rss="256K",
            req_mem="4Gn",
        )

    def test_parse_preserves_raw_oom_state(self):
        output = "12345|OUT_OF_MEMORY|0:125|00:00:05||4Gn\n"
        detail = Slurm().parse_detail(output)
        assert detail is not None
        assert detail.state == "OUT_OF_MEMORY"

    def test_parse_preserves_decorated_cancelled_state(self):
        output = "12345|CANCELLED+|0:0|00:00:03||4Gn\n"
        detail = Slurm().parse_detail(output)
        assert detail is not None
        assert detail.state == "CANCELLED+"

    def test_parse_empty_output_returns_none(self):
        assert Slurm().parse_detail("") is None
        assert Slurm().parse_detail("\n\n") is None

    def test_parse_no_parent_row_returns_none(self):
        # Only step rows present, no parent row -> cannot identify primary state.
        output = "12345.batch|COMPLETED|0:0|00:00:05|256K|4Gn\n"
        assert Slurm().parse_detail(output) is None

    def test_parse_skips_short_rows(self):
        # A malformed row (too few fields) is dropped, parent row still wins.
        output = "broken\n12345|COMPLETED|0:0|00:00:05||4Gn\n"
        detail = Slurm().parse_detail(output)
        assert detail is not None
        assert detail.state == "COMPLETED"


class TestPJMDetailNotSupported:
    def test_pjm_detail_cmd_returns_none(self):
        assert PJM().detail_cmd("12345") is None

    def test_pjm_parse_detail_returns_none(self):
        assert PJM().parse_detail("anything") is None
