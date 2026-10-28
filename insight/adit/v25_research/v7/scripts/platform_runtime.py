"""Native process locks/RSS on Linux; equivalent local CPU support on Windows."""
import os
try:
    import fcntl
    import resource
except ImportError:
    assert os.name=='nt'
    import ctypes
    from ctypes import wintypes
    import msvcrt
    from types import SimpleNamespace

    def flock(fd,operation):
        assert operation==3, 'Only exclusive nonblocking locks are used here'
        # Native fcntl accepts both an open file object and a descriptor.
        fd = fd.fileno() if hasattr(fd, 'fileno') else fd
        if os.fstat(fd).st_size==0:os.write(fd,b'0')
        os.lseek(fd,0,os.SEEK_SET)
        msvcrt.locking(fd,msvcrt.LK_NBLCK,1)

    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in [
            'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
            'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]

    def getrusage(kind):
        assert kind==0
        values=Counters();values.cb=ctypes.sizeof(values)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(values),values.cb):raise ctypes.WinError(ctypes.get_last_error())
        return SimpleNamespace(ru_maxrss=values.PeakWorkingSetSize/1024.)

    fcntl=SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=flock)
    resource=SimpleNamespace(RUSAGE_SELF=0,getrusage=getrusage)
