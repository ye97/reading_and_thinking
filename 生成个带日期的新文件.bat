rem ren %1 "%~n1-%date:~0,4%-%date:~5,2%-%date:~8,2%-%time:~0,2%-%time:~3,2%-%time:~6,2%%~x1"
rem ren重命名
rem 下面是复制一份
rem  copy %1 "%~n1-%date:~0,4%-%date:~5,2%-%date:~8,2%-%time:~0,2%-%time:~3,2%-%time:~6,2%%~x1"

copy  %1 "%~n1-%date:~0,4%-%date:~5,2%-%date:~8,2%%~x1"