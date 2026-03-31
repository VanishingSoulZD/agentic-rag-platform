from memory_store import append_message2, get_memory2



if __name__ == "__main__":
    session_id = 'abc12311'

    append_message2(session_id, {
        "role": "user",
        "content": "hello",
    })
    append_message2(session_id, {
        "role": "assistant",
        "content": "hello11",
    })
    history = get_memory2(session_id)
    print("1")
    print(history)