from typing import List, Union, Generator, Iterator

class Pipeline:
    def __init__(self):
        self.name = "AI Teaching Step 0"
        pass

    async def on_startup(self):
        print(f"[STARTUP] {self.name} pipeline started.")
        pass

    async def on_shutdown(self):
        print(f"[SHUTDOWN] {self.name} pipeline stopped.")
        pass

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"[MESSAGE] Teaching topic requested: {user_message}")
        return f"✅ I will teach you about: **{user_message}**"
