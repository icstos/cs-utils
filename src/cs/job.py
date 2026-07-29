from __future__ import annotations

from enum import Enum
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler


class JobMode(Enum):
    CRON = "cron"
    INTERVAL = "interval"

    @property
    def description(self) -> str:
        return {
            JobMode.CRON: "cron间隔性执行",
            JobMode.INTERVAL: "interval间隔性执行",
        }[self]


class Job:
    def __init__(
        self,
        timezone: str = "Asia/Shanghai",
        job_defaults: dict[str, Any] | None = None,
    ):
        self._scheduler = BackgroundScheduler(
            timezone=timezone,
            job_defaults=job_defaults or {"coalesce": False, "max_instances": 1},
        )
        self._started = False

    def add_job(
        self,
        func: Any,
        trigger: JobMode = JobMode.CRON,
        *,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        job_id: str | None = None,
        **trigger_args: Any,
    ) -> None:
        """Add a job to the scheduler.

        Args:
            func: Callable to schedule.
            trigger: JobMode.CRON or JobMode.INTERVAL.
            args: Positional arguments for the job function.
            kwargs: Keyword arguments for the job function.
            job_id: Optional job identifier.
            trigger_args: Trigger-specific scheduling parameters.
        """

        if trigger not in JobMode:
            raise ValueError(f"Unsupported trigger type: {trigger}")

        self._scheduler.add_job(
            func,
            trigger.value,
            args=args or (),
            kwargs=kwargs or {},
            id=job_id,
            **trigger_args,
        )

    def add_cron_job(self, func: Any, **cron_args: Any) -> None:
        """Add a cron-based job."""
        if not cron_args:
            cron_args = {"hour": 23, "minute": 59}
        self.add_job(func, JobMode.CRON, **cron_args)

    def add_interval_job(self, func: Any, **interval_args: Any) -> None:
        """Add an interval-based job."""
        self.add_job(func, JobMode.INTERVAL, **interval_args)

    def start(self) -> None:
        if not self._started:
            self._scheduler.start()
            self._started = True

    def stop(self, wait: bool = True) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
            self._started = False


def test_job():
    print("Job executed!")


if __name__ == "__main__":
    job = Job()

    job.add_job(test_job, JobMode.INTERVAL, minutes=2)  # 以 interval 间隔性执行
    job.add_job(test_job, JobMode.CRON, hour=23, minute=59)  # 以 cron 间隔性执行
    job.add_job(test_job, JobMode.INTERVAL, minutes=5)
    job.add_job(
        test_job,
        JobMode.CRON,
        day_of_week="*",
        hour="0",
        minute="0",
        second="0",
        id="scheduler_job1",
    )
    # sc.add_job(
    # test_job, JobMode.CRON, day_of_week='*', hour='*/2', minute='0', second='0', id='scheduler_job2'
    # )
    # # 测试任务
    job.add_job(
        test_job,
        JobMode.CRON,
        day_of_week="*",
        hour="1",
        minute="*",
        second="*",
        id="get_yolo_update",
    )
    job.add_job(
        test_job,
        JobMode.INTERVAL,
        minutes=5,
        # args=["test_arg1", "test_arg2"],
        kwargs={"key1": "value1", "key2": "value2"},
    )
    job.add_job(
        test_job,
        JobMode.CRON,
        day_of_week="*",
        hour="0",
        minute="0",
        second="0",
        id="every_day_job",
    )
    job.add_job(
        test_job,
        JobMode.CRON,
        day_of_week="1",
        hour="0",
        minute="0",
        second="0",
        id="every_week_job",
    )
    job.add_cron_job(lambda: print("Cron job executed!"))
    job.add_interval_job(lambda: print("Interval job executed!"), seconds=10)
    job.start()
