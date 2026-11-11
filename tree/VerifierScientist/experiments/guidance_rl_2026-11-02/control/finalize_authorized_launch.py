"""Withdrawn legacy approval materializer. Historical source is archived.
User requires a new reviewed configuration; this entry point never authorizes.
"""

def authorize(*args, **kwargs):
    raise RuntimeError("Automatic authorization is withdrawn; use a newly reviewed explicit launch.")

def main():
    authorize()

if __name__ == "__main__":
    main()
