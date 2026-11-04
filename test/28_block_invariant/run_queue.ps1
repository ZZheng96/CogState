Set-Location D:\BCIC2026\CogState
while (-not (Select-String -Quiet -Path test\11_individual_20261102\build_sliding1_new.log -Pattern "^\[done\]")) { Start-Sleep 30 }
foreach ($s in "raw","cat","comb") {
  $p = Start-Process python -ArgumentList "-u","test\28_block_invariant\e28main.py",$s -NoNewWindow -Wait -PassThru -RedirectStandardOutput "test\28_block_invariant\run_$s.out" -RedirectStandardError "test\28_block_invariant\run_$s.err"
  if ($p.ExitCode -ne 0) { "FAILED $s" | Out-File test\28_block_invariant\run_queue.status; exit 1 }
}
"DONE" | Out-File test\28_block_invariant\run_queue.status
