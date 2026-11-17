# Owner measured work interval — truthfully closed

| start epoch | end epoch | duration | activity | role | status |
| --- | --- | --- | --- | --- | --- |
| 1794100004 | 1794103720 | 3716s | ONE SourceB real16 + externalPOST + independent audit wait + Root disposition + final metadata/report/seal; queue/waits included, prior independent PREQUEUE271s excluded | review-agent | FINAL_FAILED_NOT_ACCEPTED |

Only this actual owner interval closed at the real final clock after the audit
and Root disposition. No parent or other-agent OPEN row was closed. Shared
writes0; Root imports work_record.json once. All owned jobs terminal.
