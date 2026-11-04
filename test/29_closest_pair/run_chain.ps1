# E29 缓存重建链 + 最接近组对聚合，随后恢复 E28 raw/cat/comb（串行，避免争显存）
Set-Location D:\BCIC2026\CogState
$st = "test\29_closest_pair\run_chain.status"
Copy-Item test\26_merge2\merge2.json test\29_closest_pair\merge2_orig.json -Force
$jobs = @(@("test\19_xblock\xblock2.py"), @("test\23_synthesis\common.py"), @("test\25_merge\pairs.py"),
          @("test\26_merge2\merge2.py"), @("test\29_closest_pair\closest.py"),
          @("test\28_block_invariant\e28main.py","raw"), @("test\28_block_invariant\e28main.py","cat"),
          @("test\28_block_invariant\e28main.py","comb"), @("test\28_block_invariant\e28.py"))
foreach ($j in $jobs) {
  $tag = [IO.Path]::GetFileNameWithoutExtension($j[0]) + $(if ($j.Count -gt 1) { "_" + $j[1] } else { "" })
  "RUNNING $tag $(Get-Date -Format HH:mm)" | Out-File $st
  $p = Start-Process python -ArgumentList (@("-u") + $j) -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "test\29_closest_pair\chain_$tag.out" -RedirectStandardError "test\29_closest_pair\chain_$tag.err"
  if ($p.ExitCode -ne 0) { "FAILED $tag" | Out-File $st; exit 1 }
}
"DONE" | Out-File $st
