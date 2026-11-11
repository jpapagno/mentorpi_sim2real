@echo off
cd /d "C:\isaacsim\isaac-sim-standalone-6.2.0-windows-x86_64"
call isaac-sim.bat --/app/vulkan=false %*
