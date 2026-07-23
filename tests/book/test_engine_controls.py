from __future__ import annotations

import asyncio

import pytest

from deeptutor.book.engine import BookEngine, _BookRuntime
from deeptutor.book.models import (
    Block,
    BlockStatus,
    BlockType,
    Book,
    Chapter,
    Page,
    PageStatus,
    Spine,
)


def test_force_compile_reset_preserves_user_notes() -> None:
    generated = Block(
        type=BlockType.CODE,
        status=BlockStatus.READY,
        payload={"code": "print(1)"},
        source_anchors=[],
        metadata={"generation_ms": 10, "transition_in": "bridge"},
    )
    note = Block(
        type=BlockType.USER_NOTE,
        status=BlockStatus.READY,
        payload={"body": "keep me"},
    )
    page = Page(status=PageStatus.READY, error="", blocks=[generated, note])

    BookEngine._reset_page_for_force_compile(page)

    assert page.status == PageStatus.PENDING
    assert generated.status == BlockStatus.PENDING
    assert generated.payload == {}
    assert generated.error == ""
    assert generated.metadata == {"transition_in": "bridge"}
    assert note.status == BlockStatus.READY
    assert note.payload == {"body": "keep me"}


class _RecordingStorage:
    """Minimal stand-in for BookStorage: records or refuses save_page calls."""

    def __init__(self, fail: bool = False):
        self.saved: list[Page] = []
        self.fail = fail

    def save_page(self, page: Page) -> None:
        if self.fail:
            raise OSError("disk full")
        self.saved.append(page)


def _engine_with_storage(storage: _RecordingStorage) -> BookEngine:
    engine = BookEngine.__new__(BookEngine)
    engine.storage = storage
    engine._global_lock = asyncio.Lock()
    return engine


def test_mark_page_error_resets_generating_page() -> None:
    storage = _RecordingStorage()
    engine = _engine_with_storage(storage)
    page = Page(status=PageStatus.GENERATING)

    engine._mark_page_error(page, RuntimeError("llm timeout"), prefix="Compilation failed")

    assert page.status == PageStatus.ERROR
    assert "llm timeout" in page.error
    assert storage.saved == [page]


def test_mark_page_error_resets_planning_page() -> None:
    storage = _RecordingStorage()
    engine = _engine_with_storage(storage)
    page = Page(status=PageStatus.PLANNING)

    engine._mark_page_error(page, RuntimeError("planner crashed"), prefix="Compilation failed")

    assert page.status == PageStatus.ERROR
    assert "planner crashed" in page.error
    assert storage.saved == [page]


def test_mark_page_error_ignores_missing_or_settled_pages() -> None:
    storage = _RecordingStorage()
    engine = _engine_with_storage(storage)

    engine._mark_page_error(None, RuntimeError("boom"), prefix="x")
    ready = Page(status=PageStatus.READY)
    engine._mark_page_error(ready, RuntimeError("boom"), prefix="x")

    assert storage.saved == []
    assert ready.status == PageStatus.READY


def test_mark_page_error_survives_save_failure() -> None:
    engine = _engine_with_storage(_RecordingStorage(fail=True))
    page = Page(status=PageStatus.GENERATING)

    # Runs inside exception handlers (worker loop) — must never raise.
    engine._mark_page_error(page, RuntimeError("boom"), prefix="x")

    assert page.status == PageStatus.ERROR


class _SpinePageStorage:
    """Minimal storage double for page-to-spine consistency tests."""

    def __init__(self, pages: list[Page], spine: Spine) -> None:
        self.pages = {page.id: page for page in pages}
        self.spine = spine
        self.book = Book(id=spine.book_id)
        self.deleted: list[str] = []
        self.logs: list[str] = []

    def load_book(self, book_id: str) -> Book | None:
        return self.book if book_id == self.book.id else None

    def save_book(self, book: Book) -> None:
        self.book = book

    def list_pages(self, book_id: str) -> list[Page]:
        return list(self.pages.values())

    def load_page(self, book_id: str, page_id: str) -> Page | None:
        return self.pages.get(page_id)

    def load_spine(self, book_id: str) -> Spine:
        return self.spine

    def save_spine(self, spine: Spine) -> None:
        self.spine = spine

    def save_page(self, page: Page) -> None:
        self.pages[page.id] = page

    def delete_page(self, book_id: str, page_id: str) -> bool:
        self.deleted.append(page_id)
        return self.pages.pop(page_id, None) is not None

    def append_log(self, book_id: str, message: str, *, op: str = "info") -> None:
        self.logs.append(f"{op}:{message}")


def _engine_with_spine_storage(storage: _SpinePageStorage) -> BookEngine:
    engine = BookEngine.__new__(BookEngine)
    engine.storage = storage
    engine._runtimes = {}
    engine._global_lock = asyncio.Lock()
    return engine


def test_engine_hides_and_cleans_pages_removed_from_spine() -> None:
    current_chapter = Chapter(id="ch_current")
    orphan = Page(id="pg_orphan", book_id="bk_test", chapter_id="ch_removed")
    current = Page(id="pg_current", book_id="bk_test", chapter_id=current_chapter.id)
    storage = _SpinePageStorage(
        pages=[orphan, current],
        spine=Spine(book_id="bk_test", chapters=[current_chapter]),
    )
    engine = _engine_with_spine_storage(storage)

    assert [page.id for page in engine.list_pages("bk_test")] == ["pg_current"]
    assert engine.load_page("bk_test", "pg_orphan") is None

    removed = engine._remove_orphan_pages("bk_test", storage.spine)

    assert removed == ["pg_orphan"]
    assert storage.deleted == ["pg_orphan"]
    assert "removed 1 orphan book page(s)" in storage.logs[0]


@pytest.mark.asyncio
async def test_confirm_spine_removes_pages_from_deleted_chapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_chapter = Chapter(id="ch_current")
    orphan = Page(id="pg_orphan", book_id="bk_test", chapter_id="ch_removed")
    current = Page(id="pg_current", book_id="bk_test", chapter_id=current_chapter.id)
    storage = _SpinePageStorage(
        pages=[orphan, current],
        spine=Spine(book_id="bk_test", chapters=[current_chapter]),
    )
    engine = _engine_with_spine_storage(storage)

    async def keep_spine(spine: Spine, *_: object, **__: object) -> Spine:
        return spine

    async def skip_overview_materialization(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr(engine, "_ensure_overview_chapter", keep_spine)
    monkeypatch.setattr(engine, "_materialize_overview_page", skip_overview_materialization)

    pages = await engine.confirm_spine(
        book_id="bk_test",
        edited_spine=Spine(book_id="bk_test", chapters=[current_chapter]),
        auto_compile=False,
    )

    assert [page.id for page in pages] == ["pg_current"]
    assert storage.deleted == ["pg_orphan"]
    assert list(storage.pages) == ["pg_current"]


@pytest.mark.asyncio
async def test_confirm_spine_cancels_foreground_compile_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_chapter = Chapter(id="ch_removed")
    current_chapter = Chapter(id="ch_current")
    orphan = Page(id="pg_orphan", book_id="bk_test", chapter_id=old_chapter.id)
    current = Page(id="pg_current", book_id="bk_test", chapter_id=current_chapter.id)
    storage = _SpinePageStorage(
        pages=[orphan, current],
        spine=Spine(book_id="bk_test", chapters=[old_chapter, current_chapter]),
    )
    engine = _engine_with_spine_storage(storage)
    started = asyncio.Event()

    class _BlockingCompiler:
        async def compile_page(self, **_: object) -> Page:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    engine.compiler = _BlockingCompiler()

    async def keep_spine(spine: Spine, *_: object, **__: object) -> Spine:
        return spine

    async def skip_overview_materialization(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr(engine, "_ensure_overview_chapter", keep_spine)
    monkeypatch.setattr(engine, "_materialize_overview_page", skip_overview_materialization)

    foreground = asyncio.create_task(engine.compile_page(book_id="bk_test", page_id="pg_orphan"))
    await started.wait()

    pages = await engine.confirm_spine(
        book_id="bk_test",
        edited_spine=Spine(book_id="bk_test", chapters=[current_chapter]),
        auto_compile=False,
    )

    with pytest.raises(asyncio.CancelledError):
        await foreground
    assert [page.id for page in pages] == ["pg_current"]
    assert storage.deleted == ["pg_orphan"]
    assert list(storage.pages) == ["pg_current"]


@pytest.mark.asyncio
async def test_reconfiguration_waits_for_running_compile_to_exit() -> None:
    engine = _engine_with_storage(_RecordingStorage())
    started = asyncio.Event()
    stopped = asyncio.Event()
    never_set = asyncio.Event()

    async def worker() -> None:
        started.set()
        try:
            await never_set.wait()
        finally:
            stopped.set()

    task = asyncio.create_task(worker())
    await started.wait()
    runtime = _BookRuntime(worker=task)
    runtime.queued.add("pg_old")
    await runtime.queue.put("pg_old")
    engine._runtimes = {"bk_test": runtime}

    await engine._begin_book_reconfiguration("bk_test")

    assert task.cancelled()
    assert stopped.is_set()
    assert runtime.worker is None
    assert runtime.queued == set()
    assert runtime.queue.empty()

    await engine._end_book_reconfiguration("bk_test")
    assert runtime.reconfiguration_depth == 0
