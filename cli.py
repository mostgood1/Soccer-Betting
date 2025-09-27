"""Simple CLI wrapper for offline tasks.
Examples:
  python cli.py retrain --level patch
  python cli.py rebuild
  python cli.py summary --week 1
"""
import sys
from app.offline.tasks import main as offline_main

if __name__ == '__main__':
    offline_main()
