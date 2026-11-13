"""Convenience entry point for the cricket_coach package."""

from cricket_coach.run_pipeline import main


if __name__ == "__main__":
    assert callable(main)
    main()
