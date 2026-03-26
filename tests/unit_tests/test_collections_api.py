from uuid import UUID

from tests.unit_tests.fixtures import get_async_test_client


async def test_health() -> None:
    """Test the health check endpoint."""
    async with get_async_test_client() as client:
        response = await client.get("/health")
        response.raise_for_status()
        assert response.json() == {"status": "ok"}


async def test_create_and_get_collection() -> None:
    """Test creating and retrieving a collection."""
    async with get_async_test_client() as client:
        payload = {"name": "test_collection", "metadata": {"purpose": "unit-test"}}
        response = await client.post(
            "/collections", json=payload
        )
        assert response.status_code == 201, (
            f"Failed with error message: {response.text}"
        )
        data = response.json()
        assert data["name"] == "test_collection"
        assert isinstance(UUID(data["uuid"]), UUID)

        # Get collection by ID
        get_response = await client.get(
            f"/collections/{data['uuid']}"
        )
        assert get_response.status_code == 200
        assert get_response.json()["uuid"] == data["uuid"]

        # Test without metadata
        payload_no_metadata = {"name": "test_collection_no_metadata"}
        response_no_metadata = await client.post(
            "/collections", json=payload_no_metadata
        )
        assert response_no_metadata.status_code == 201, (
            f"Failed with error message: {response_no_metadata.text}"
        )
        data_no_metadata = response_no_metadata.json()
        assert data_no_metadata == {
            "uuid": data_no_metadata["uuid"],
            "name": "test_collection_no_metadata",
            "metadata": {},
        }


async def test_create_and_list_collection_with_stats() -> None:
    """Test creating and listing a collection with stats."""
    async with get_async_test_client() as client:
        payload = {"name": "test_collection_with_stats", "metadata": {"purpose": "unit-test"}}
        response = await client.post(
            "/collections", json=payload
        )
        assert response.status_code == 201, (
            f"Failed with error message: {response.text}"
        )
        data = response.json()
        assert data["name"] == "test_collection_with_stats"
        assert isinstance(UUID(data["uuid"]), UUID)

        # List collections and check for stats
        list_response = await client.get("/collections")
        assert list_response.status_code == 200
        collections = list_response.json()
        assert len(collections) > 0

        test_collection = next((c for c in collections if c["name"] == "test_collection_with_stats"), None)
        assert test_collection is not None
        assert "document_count" in test_collection
        assert "chunk_count" in test_collection
        assert test_collection["document_count"] == 0
        assert test_collection["chunk_count"] == 0



async def test_create_collections_with_identical_names() -> None:
    """Test that collections with identical names can be created."""
    async with get_async_test_client() as client:
        payload = {"name": "dup_collection", "metadata": {"foo": "bar"}}
        # first create
        r1 = await client.post("/collections", json=payload)
        assert r1.status_code == 201

        # second create with same name
        r2 = await client.post("/collections", json=payload)
        assert r2.status_code == 201
        assert r1.json()["uuid"] != r2.json()["uuid"]


async def test_get_nonexistent_collection() -> None:
    """GET a collection that doesn't exist should be 404."""
    async with get_async_test_client() as client:
        r = await client.get("/collections/nonexistent")
        # Not a UUID, so should be 422
        assert r.status_code == 422

        r = await client.get(
            "/collections/12345678-1234-5678-1234-567812345678"
        )
        assert r.status_code == 404


async def test_delete_collection_and_nonexistent() -> None:
    """DELETE removes an existing collection and returns 404 on missing."""
    async with get_async_test_client() as client:
        # create first
        payload = {"name": "to_delete", "metadata": {"foo": "bar"}}
        r1 = await client.post("/collections", json=payload)
        assert r1.status_code == 201

        # Get the UUID first
        get_collection = await client.get("/collections")
        assert get_collection.status_code == 200
        collections = get_collection.json()
        collection_id = next(
            (c["uuid"] for c in collections if c["name"] == "to_delete"), None
        )
        assert collection_id is not None

        # delete it by ID
        r2 = await client.delete(
            f"/collections/{collection_id}"
        )
        assert r2.status_code == 204

        # Try to get it again by ID
        r3 = await client.get(f"/collections/{collection_id}")
        assert r3.status_code == 404

        # Deletion is idempotent
        r4 = await client.delete(
            f"/collections/{collection_id}"
        )
        assert r4.status_code == 204


async def test_patch_collection() -> None:
    """PATCH should update metadata properly."""
    async with get_async_test_client() as client:
        # create a collection
        payload = {"name": "colA", "metadata": {"a": 1}}
        r = await client.post("/collections", json=payload)
        assert r.status_code == 201
        assert r.json() == {
            "uuid": r.json()["uuid"],
            "name": "colA",
            "metadata": {
                "a": 1,
            },
        }

        # Get the UUID for colA
        get_collection = await client.get("/collections")
        assert get_collection.status_code == 200
        collections = get_collection.json()
        collection_id = next(
            (c["uuid"] for c in collections if c["name"] == "colA"), None
        )
        assert collection_id is not None

        # update metadata using the UUID
        r2 = await client.patch(
            f"/collections/{collection_id}",
            json={"metadata": {"a": 2}},
        )
        assert r2.status_code == 200
        assert r2.json() == {
            "uuid": r.json()["uuid"],
            "name": "colA",
            "metadata": {
                "a": 2,
            },
        }


async def test_update_collection_name_and_metadata() -> None:
    """PATCH should rename and/or update metadata properly."""
    async with get_async_test_client() as client:
        # create two collections
        await client.post(
            "/collections",
            json={"name": "colA", "metadata": {"a": 1}},
        )
        await client.post(
            "/collections",
            json={"name": "colB", "metadata": {"b": 2}},
        )

        # Get the UUID for colA
        get_collection = await client.get("/collections")
        assert get_collection.status_code == 200
        collections = get_collection.json()
        col_a_id = next((c["uuid"] for c in collections if c["name"] == "colA"), None)
        assert col_a_id is not None

        # try renaming colA to colB (conflict)
        no_conflict = await client.patch(
            f"/collections/{col_a_id}",
            json={"name": "colB"},
        )
        assert no_conflict.status_code == 200
        assert no_conflict.json() == {
            "metadata": {
                "a": 1,
            },
            "name": "colB",
            "uuid": col_a_id,  # The ID should not change
        }

        # rename colA to colC with new metadata (using the UUID we got earlier)
        update = await client.patch(
            f"/collections/{col_a_id}",
            json={"name": "colC", "metadata": {"x": "y"}},
        )
        assert update.status_code == 200
        body = update.json()
        assert body == {
            "uuid": body["uuid"],
            "name": "colC",
            "metadata": {"x": "y"},
        }
        # ensure we can get by the ID
        get_by_id = await client.get(f"/collections/{col_a_id}")
        assert get_by_id.status_code == 200
        # the ID should remain the same even though name changed
        assert get_by_id.json()["uuid"] == col_a_id
        assert get_by_id.json()["name"] == "colC"

        # update metadata only on colC using the same ID
        meta_update = await client.patch(
            f"/collections/{col_a_id}",
            json={"metadata": {"foo": "bar"}},
        )
        assert meta_update.status_code == 200
        assert meta_update.json() == {
            "uuid": col_a_id,
            "name": "colC",
            "metadata": {"foo": "bar"},
        }


async def test_update_nonexistent_collection() -> None:
    """PATCH a missing collection should return 404."""
    async with get_async_test_client() as client:
        r = await client.patch(
            "/collections/does_not_exist",
            json={"metadata": {"any": "thing"}},
        )
        assert r.status_code == 422

        r = await client.patch(
            "/collections/12345678-1234-5678-1234-567812345678",
            json={"metadata": {"any": "thing"}},
        )
        assert r.status_code == 404


async def test_list_empty_and_multiple_collections() -> None:
    """Listing when empty and after multiple creates."""
    async with get_async_test_client() as client:
        # ensure database is empty
        empty = await client.get(
            "/collections",
        )
        assert empty.status_code == 200
        assert empty.json() == []

        # create several
        names = ["one", "two", "three"]
        for n in names:
            r = await client.post(
                "/collections", json={"name": n, "metadata": {}}
            )
            assert r.status_code == 201

        listed = await client.get("/collections")
        assert listed.status_code == 200
        got = [c["name"] for c in listed.json()]
        for n in names:
            assert n in got
