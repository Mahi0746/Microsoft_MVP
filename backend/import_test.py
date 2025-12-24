import traceback
import sys
sys.path.insert(0, r'C:\Microsoft_Hackathon\new\Microsoft_MVP\backend')
try:
    import main
    print('Imported main successfully')
except Exception:
    traceback.print_exc()
    raise
