import asyncio
from app.service.video import convert_posetovideo

if __name__ == "__main__":
    t= asyncio.run(convert_posetovideo("/workspace/signlang/output_720.mp4"))
    print(t)