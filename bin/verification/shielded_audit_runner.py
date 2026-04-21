import sys
import os
import io

# Absolute Environmental Shield: Forcing Non-Interactive Mode
# This script monkey-patches stdin and sets headless environment variables
# to bypass the "Press RETURN" hangs in core dependencies.

# 1. Environment Hardening
os.environ['TERM'] = 'dumb'
os.environ['PAGER'] = 'cat'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DEBIAN_FRONTEND'] = 'noninteractive'

# 2. Monkey-patching stdin to automatically Return
class DummyStdin(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__("\n\n\n\n\n\n")  # Provide multiple newlines
    def fileno(self):
        return 0 # Pretend we are stdin

sys.stdin = DummyStdin()

# 3. Executing the target script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 shielded_audit_runner.py <script.py> [args]")
        sys.exit(1)
        
    target_script = sys.argv[1]
    # Set sys.argv for the target script
    sys.argv = sys.argv[1:]
    
    # Pre-import to verify the sink
    print(f"--- Launching Shielded Execution: {target_script} ---")
    
    # Absolute Import Bridge (Force non-interactive)
    import iriscc
    
    # Execute the target
    with open(target_script, 'r') as f:
        exec(f.read(), {'__name__': '__main__', '__file__': target_script})
