import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "gtosolver.api:app",
        host=os.environ.get("GTOSOLVER_HOST", "127.0.0.1"),
        port=int(os.environ.get("GTOSOLVER_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
