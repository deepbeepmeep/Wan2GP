// 2>nul||@goto :batch
/*
:batch
@echo off
setlocal

:: find csc.exe
set "csc="
for /r "%SystemRoot%\Microsoft.NET\Framework\" %%# in ("*csc.exe") do  set "csc=%%#"

if not exist "%csc%" (
   echo no .net framework installed
   exit /b 10
)

if not exist "%~n0.exe" (
   call %csc% /nologo /w:0 /out:"%~n0.exe" "%~dpsfnx0" || (
      exit /b %errorlevel% 
   )
)
%~n0.exe %*
endlocal & exit /b %errorlevel%

*/

using System;
using System.Runtime.InteropServices;


class QE
{
    //https://msdn.microsoft.com/en-us/library/windows/desktop/ms686033(v=vs.85).aspx
    //http://stackoverflow.com/questions/6828450/console-application-with-selectable-text

    [DllImport("kernel32.dll")]
    static extern bool SetConsoleMode(IntPtr hConsoleHandle, int mode);
    [DllImport("kernel32.dll")]
    static extern bool GetConsoleMode(IntPtr hConsoleHandle, out int mode);
    [DllImport("kernel32.dll")]
    static extern IntPtr GetStdHandle(int handle);

    const int STD_INPUT_HANDLE = -10;
    const int ENABLE_QUICK_EDIT_MODE = 0x40 | 0x80;
    const int DISABLE_QUICK_EDIT_MODE = 0x0080;
    const int CHECK_QUICK_EDIT = 0x0040;

    static void printHelp()
    {
        //clears the extension from the script name
        String scriptName = Environment.GetCommandLineArgs()[0];
        scriptName = scriptName.Substring(0, scriptName.Length-4);
        Console.WriteLine(scriptName + " enables or disables quick edit mode");
        Console.WriteLine("");
        Console.WriteLine("Enable:");
        Console.WriteLine(" " + scriptName + "  1");
        Console.WriteLine("Disable:");
        Console.WriteLine(" " + scriptName + "  2");
        Console.WriteLine("Get State:");
        Console.WriteLine(" " + scriptName + "  3");
    }

    static void Main(string[] args)
    {
        if (args.Length != 1) {
            printHelp();
            Environment.Exit(1);
        }
        if (args[0].ToLower() == "-h") {
            printHelp();
            Environment.Exit(0);
        }
        int sw;
        try
        {
            int mode;
            IntPtr handle = GetStdHandle(STD_INPUT_HANDLE);
            GetConsoleMode(handle, out mode);
            sw = int.Parse(args[0]);
            switch (sw) {
                case 1:
					//enable
                    SetConsoleMode(handle, mode|(ENABLE_QUICK_EDIT_MODE));
                    break;
                case 2:
				    //disable
                    SetConsoleMode(handle, mode &= ~CHECK_QUICK_EDIT);
                    break;
                case 3:
				    // get state
                    if (mode == (mode | ENABLE_QUICK_EDIT_MODE))
                    {
                        Console.WriteLine("enabled");
                    }
                    else {
                        Console.WriteLine("disabled");
                    }
                    break;
                default:
                    Console.WriteLine("Invalid argument: "+args[0]);
                    Environment.Exit(10);
                    break;
            }

        }
        catch (Exception e) {
            Console.WriteLine("Problem while parsing state " + args[0]);
            Environment.Exit(5);
        }

    }
}
