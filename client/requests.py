import httpx


async def send_request(d: dict, url: str):
    """Send an asynchronous POST request to the server and return the response."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=d)
            if response.status_code == 200:
                return response
            else:
                raise Exception(f"Error: {response.status_code} {response.text}")
    except httpx.HTTPStatusError as e:
        raise Exception(f"Error: {e.response.status_code} {e.response.text}")
    except httpx.RequestError as e:
        raise Exception(f"Error: Unable to send request - {str(e)}")
