# 在 e28main 队列完成后，依次重建 fuse.py 需要的分数缓存（避免与 e28main 争 6GB 显存）
Set-Location D:\BCIC2026\CogState
while (-not (Test-Path test\28_block_invariant\run_queue.status)) { Start-Sleep 60 }
$jobs = @(@("test\19_xblock\xblock2.py","test\19_xblock\xblock2_rerun"),
          @("test\23_synthesis\common.py","test\23_synthesis\common_rerun"),
          @("test\28_block_invariant\e28.py","test\28_block_invariant\e28_rerun"))
foreach ($j in $jobs) {
  $p = Start-Process python -ArgumentList "-u",$j[0] -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$($j[1]).log" -RedirectStandardError "$($j[1]).err"
  if ($p.ExitCode -ne 0) { "FAILED $($j[0])" | Out-File test\28_block_invariant\run_fuse_deps.status; exit 1 }
}
"DONE" | Out-File test\28_block_invariant\run_fuse_deps.status
