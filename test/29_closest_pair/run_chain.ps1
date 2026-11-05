# E29 最接近组对聚合，随后 E28 cat/comb（缓存与 raw 已从旧机器拷入；串行，避免争显存）
# 启动须用 WMI（Invoke-CimMethod Win32_Process Create），否则随终端命令结束被杀
Set-Location D:\BCIC2026\CogState
$st = "test\29_closest_pair\run_chain.status"
$jobs = @(@("test\29_closest_pair\closest.py"),
          @("test\28_block_invariant\e28main.py","cat"),
          @("test\28_block_invariant\e28main.py","comb"), @("test\28_block_invariant\e28.py"))
foreach ($j in $jobs) {
  $tag = [IO.Path]::GetFileNameWithoutExtension($j[0]) + $(if ($j.Count -gt 1) { "_" + $j[1] } else { "" })
  "RUNNING $tag $(Get-Date -Format HH:mm)" | Out-File $st
  $p = Start-Process python -ArgumentList (@("-u") + $j) -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "test\29_closest_pair\chain_$tag.out" -RedirectStandardError "test\29_closest_pair\chain_$tag.err"
  if ($p.ExitCode -ne 0) { "FAILED $tag" | Out-File $st; exit 1 }
}
"DONE" | Out-File $st
