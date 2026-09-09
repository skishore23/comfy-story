from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import pytest

from duet.duetx.story_contracts import (
    DuetStoryStateRef,
    ReferenceRole,
    StoryLibrary,
    StoryReference,
    canonical_story_json,
)
from duet.duetx.story_native_archive import (
    NATIVE_REFERENCE_RUNTIME_SHA256,
    NativeArchiveRevision,
    NativeReferenceArchive,
    NativeReferenceReceipt,
)
from duet.duetx.story_product_contracts import (
    CanonEntity,
    CanonPresence,
    GuideBinding,
    NativeRGBObservation,
    ObservationKind,
    StoryCanon,
    StoryEvidenceRecord,
    StoryMemoryPolicy,
    StoryObservation,
    StoryObservationPacket,
    StoryProductState,
)
from duet.duetx.story_store import StoryProjectStore


def _setup(tmp_path: Path) -> tuple[StoryProjectStore, NativeReferenceArchive, StoryLibrary]:
    assets = StoryProjectStore(tmp_path)
    baseline = assets.put_asset(b"reference image")
    library = StoryLibrary(
        "Oak",
        (
            StoryReference(
                "Acorn",
                ReferenceRole.PROP,
                "Brown shell",
                baseline,
                f"duet-story://assets/sha256/{baseline}",
                (baseline,),
                "1" * 64,
            ),
        ),
    ).validate()
    return assets, NativeReferenceArchive(assets), library


class NativePayload(TypedDict):
    project_id: str
    branch_id: str
    parent: DuetStoryStateRef | None
    library: StoryLibrary
    product_state: StoryProductState
    model_configuration_sha256: str
    last_frame_png: bytes
    shot_metadata: dict[str, object]
    receipt: NativeReferenceReceipt


def _payload(
    assets: StoryProjectStore, library: StoryLibrary, previous: NativeArchiveRevision | None = None
) -> NativePayload:
    index = 0 if previous is None else previous.state.shot_count
    frame = assets.put_asset(f"frame-{index}".encode())
    video = assets.put_asset(f"video-{index}".encode())
    tensor = assets.put_asset(f"tensor-{index}".encode())
    execution = assets.put_asset(
        canonical_story_json({"seed": index, "guides": [["current", frame]]})
    )
    observation = NativeRGBObservation(
        f"rgb-{index}",
        ObservationKind.CLOSING,
        0,
        index * 100,
        (index + 1) * 100,
        (0, 0, 65536, 65536),
        frame,
        "2" * 64,
        0,
        0,
        ("acorn",),
    ).validate()
    packet = StoryObservationPacket(
        f"shot-{index}",
        index,
        1,
        index * 100,
        (index + 1) * 100,
        video,
        f"duet-evidence://story/sha256/{video}",
        "3" * 64,
        ("acorn",),
        (observation,),
    ).validate()
    canon = (
        StoryCanon(
            (
                CanonEntity(
                    "acorn",
                    "Acorn",
                    library.references[0].media_sha256,
                    "Brown shell",
                    CanonPresence.UNKNOWN,
                    (),
                    None,
                    (),
                ),
            )
        )
        if previous is None
        else previous.product_state.canon
    )
    product = StoryProductState(
        (packet,) if previous is None else (*previous.product_state.observation_packets, packet),
        tuple(
            sorted(
                (StoryEvidenceRecord.from_observation(packet, observation),)
                if previous is None
                else (
                    *previous.product_state.evidence_records,
                    StoryEvidenceRecord.from_observation(packet, observation),
                ),
                key=lambda x: x.evidence_id,
            )
        ),
        canon,
        StoryMemoryPolicy(),
    ).validate()
    return {
        "project_id": "oak",
        "branch_id": "main",
        "parent": None if previous is None else previous.state,
        "library": library,
        "product_state": product,
        "model_configuration_sha256": "4" * 64,
        "last_frame_png": assets.load_asset(frame),
        "shot_metadata": {
            "video_sha256": video,
            "last_frame_full_sha256": frame,
            "last_frame_tensor_sha256": tensor,
            "execution_sha256": execution,
        },
        "receipt": NativeReferenceReceipt(
            execution, "5" * 64, (), (GuideBinding("current", frame),)
        ),
    }


def test_native_archive_reopens_exact_state_without_checkpoint_or_operators(tmp_path: Path) -> None:
    assets, archive, library = _setup(tmp_path)
    payload = _payload(assets, library)
    state = archive.publish(**payload)
    loaded = NativeReferenceArchive(StoryProjectStore(tmp_path)).load(
        state, model_configuration_sha256="4" * 64
    )
    assert loaded.product_state == payload["product_state"]
    assert loaded.last_frame_png == payload["last_frame_png"]
    assert archive.publish(**payload) == state
    assert not hasattr(loaded, "blocks")
    assert not hasattr(loaded, "memory")
    manifest = assets.load_asset(state.revision_sha256)
    assert b"checkpoint" not in manifest
    assert b"operator" not in manifest
    assert loaded.generation_receipt.runtime_sha256 == NATIVE_REFERENCE_RUNTIME_SHA256
    assert not list((tmp_path / "snapshots/sha256").iterdir())


def test_native_rgb_wire_cannot_impersonate_encoded_observation(tmp_path: Path) -> None:
    assets, _, library = _setup(tmp_path)
    product = _payload(assets, library)["product_state"]
    encoded = product.to_json()
    assert b"vae_sha256" not in encoded
    assert b"latent_sha256" not in encoded
    assert StoryProductState.from_json(encoded) == product
    value = json.loads(encoded)["observation_packets"][0]["observations"][0]
    with pytest.raises(ValueError, match="missing or unknown"):
        StoryObservation.from_mapping(value)
    value["vae_sha256"] = "6" * 64
    with pytest.raises(ValueError, match="missing or unknown"):
        NativeRGBObservation.from_mapping(value)


def test_native_append_retains_approved_canon_and_rejects_rewritten_history(tmp_path: Path) -> None:
    assets, archive, library = _setup(tmp_path)
    first = archive.publish(**_payload(assets, library))
    previous = archive.load(first, model_configuration_sha256="4" * 64)
    payload = _payload(assets, library, previous)
    product = payload["product_state"]
    entity = replace(
        product.canon.entities[0],
        state_note="Blue painted shell",
        presence=CanonPresence.PRESENT,
        supporting_evidence_ids=("rgb-0",),
        confirmed_revision_sha256=first.revision_sha256,
    )
    payload["product_state"] = replace(
        product, canon=StoryCanon((entity,)), policy=StoryMemoryPolicy(("rgb-0",), ())
    )
    payload["receipt"] = replace(payload["receipt"], selected_evidence_ids=("rgb-0",))
    second = archive.publish(**payload)
    assert second.shot_count == 2
    assert second.parent_revision_sha256 == first.revision_sha256
    loaded = archive.load(second, model_configuration_sha256="4" * 64)
    assert loaded.product_state.canon.entities[0] == entity
    assert loaded.product_state.policy.pinned_evidence_ids == ("rgb-0",)
    assert (
        archive.load(first, model_configuration_sha256="4" * 64).product_state
        == previous.product_state
    )
    altered = replace(product.observation_packets[0], extraction_policy_sha256="7" * 64)
    payload["product_state"] = replace(
        product, observation_packets=(altered, product.observation_packets[1])
    )
    with pytest.raises(ValueError, match="changed historical"):
        archive.publish(**payload)


def test_native_archive_rejects_identity_corruption_and_future_recall(tmp_path: Path) -> None:
    assets, archive, library = _setup(tmp_path)
    payload = _payload(assets, library)
    future = payload.copy()
    future["receipt"] = replace(payload["receipt"], selected_evidence_ids=("rgb-0",))
    with pytest.raises(ValueError, match="unavailable before"):
        archive.publish(**future)
    with pytest.raises(ValueError, match="runtime identity"):
        replace(payload["receipt"], runtime_sha256="0" * 64).validate()
    state = archive.publish(**payload)
    with pytest.raises(ValueError, match="different model"):
        archive.load(state, model_configuration_sha256="8" * 64)
    with pytest.raises(ValueError, match="state reference"):
        archive.load(replace(state, branch_id="other"), model_configuration_sha256="4" * 64)
    (tmp_path / "assets/sha256" / state.last_frame_sha256).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        archive.load(state, model_configuration_sha256="4" * 64)


def test_native_archive_refuses_unrelated_legacy_manifest(tmp_path: Path) -> None:
    assets, archive, _ = _setup(tmp_path)
    sha = assets.put_asset(canonical_story_json({"format": "duet-story-revision-v3"}))
    state = DuetStoryStateRef(
        "oak", "main", None, sha, "1" * 64, "2" * 64, 1, 1, "3" * 64, "4" * 64
    )
    with pytest.raises(ValueError, match="missing or unknown"):
        archive.load(state, model_configuration_sha256="4" * 64)


def test_native_revision_digest_reopens_authenticated_state(tmp_path: Path) -> None:
    assets, archive, library = _setup(tmp_path)
    state = archive.publish(**_payload(assets, library))
    assert archive.load_revision(state.revision_sha256).state == state
    with pytest.raises(ValueError, match="project, branch and integer"):
        archive.load_revision(assets.put_asset(b'{"project_id":1}'))


def test_film_state_recall_binds_selected_ending_without_approving_it(tmp_path: Path) -> None:
    from duet.duetx.film_audit import ShotVisualAudit
    from duet.duetx.film_plan import FilmFact, FilmPlan, FilmShot
    from duet.duetx.film_selected_state import SelectedStateSource, select_film_state_evidence

    assets, archive, library = _setup(tmp_path)
    first = archive.publish(**_payload(assets, library))
    loaded = archive.load_revision(first.revision_sha256)
    second = archive.publish(**_payload(assets, library, loaded))
    plan = FilmPlan(
        "oak",
        "Cracked shell",
        15000,
        (),
        (
            FilmShot(
                "crack",
                5000,
                "Change shell",
                "The shell cracks.",
                ("Acorn",),
                effects=(FilmFact("Acorn.shell", "cracked"),),
            ),
            FilmShot("cut", 5000, "Show nearby clearing", "Wind moves grass.", ()),
            FilmShot(
                "return",
                5000,
                "Return to shell",
                "The cracked shell rests on a new surface.",
                ("Acorn",),
                requires=(FilmFact("Acorn.shell", "cracked"),),
            ),
        ),
    ).validate()
    selected = tuple(
        SelectedStateSource(
            state.revision_sha256,
            "a" * 64,
            ShotVisualAudit(
                shot.shot_id,
                shot.digest,
                str(archive.load_revision(state.revision_sha256).shot_metadata["video_sha256"]),
                "b" * 64,
                (),
                tuple((f.key, f.value) for f in shot.effects),
                "machine_pass",
            ),
        )
        for shot, state in zip(plan.shots[:2], (first, second), strict=True)
    )
    bindings = select_film_state_evidence(plan, 2, selected, archive)
    assert len(bindings) == 1
    assert bindings[0].entity_id == "acorn"
    assert bindings[0].evidence_id == "rgb-0"
    assert bindings[0].source_revision_sha256 == first.revision_sha256
    assert bindings[0].assessment_sha256 == "a" * 64
    assert bindings[0].facts == (("Acorn.shell", "cracked"),)
    assert (
        archive.load_revision(second.revision_sha256).product_state.canon
        == loaded.product_state.canon
    )
    with pytest.raises(ValueError, match="passing assessments"):
        select_film_state_evidence(
            plan,
            2,
            (replace(selected[0], audit=replace(selected[0].audit, status="fail")), selected[1]),
            archive,
        )
    with pytest.raises(ValueError, match="videos do not match"):
        select_film_state_evidence(
            plan,
            2,
            (
                replace(selected[0], audit=replace(selected[0].audit, video_sha256="f" * 64)),
                selected[1],
            ),
            archive,
        )
    with pytest.raises(ValueError, match="revision lineage"):
        select_film_state_evidence(
            plan,
            2,
            (replace(selected[0], revision_sha256=second.revision_sha256), selected[1]),
            archive,
        )


def test_film_state_recall_uses_latest_joint_snapshot_and_rejects_tombstones(
    tmp_path: Path,
) -> None:
    from duet.duetx.film_audit import ShotVisualAudit
    from duet.duetx.film_plan import FilmFact, FilmPlan, FilmShot
    from duet.duetx.film_selected_state import SelectedStateSource, select_film_state_evidence

    assets, archive, library = _setup(tmp_path)
    first = archive.publish(**_payload(assets, library))
    loaded = archive.load_revision(first.revision_sha256)
    payload = _payload(assets, library, loaded)
    second = archive.publish(**payload)
    plan = FilmPlan(
        "oak",
        "Visible state changes",
        15000,
        (),
        (
            FilmShot(
                "crack",
                5000,
                "Crack",
                "Shell cracks.",
                ("Acorn",),
                effects=(FilmFact("Acorn.shell", "cracked"),),
            ),
            FilmShot(
                "wet",
                5000,
                "Wet",
                "Rain wets the shell.",
                ("Acorn",),
                effects=(FilmFact("Acorn.surface", "wet"),),
            ),
            FilmShot(
                "return",
                5000,
                "Return",
                "Wet cracked shell on a new surface.",
                ("Acorn",),
                requires=(FilmFact("Acorn.shell", "cracked"), FilmFact("Acorn.surface", "wet")),
            ),
        ),
    ).validate()

    def sources(last: DuetStoryStateRef) -> tuple[SelectedStateSource, ...]:
        return tuple(
            SelectedStateSource(
                state.revision_sha256,
                "a" * 64,
                ShotVisualAudit(
                    shot.shot_id,
                    shot.digest,
                    str(archive.load_revision(state.revision_sha256).shot_metadata["video_sha256"]),
                    "b" * 64,
                    (),
                    (),
                    "machine_pass",
                ),
            )
            for shot, state in zip(plan.shots[:2], (first, last), strict=True)
        )

    bindings = select_film_state_evidence(plan, 2, sources(second), archive)
    assert bindings[0].evidence_id == "rgb-1"
    assert bindings[0].facts == (("Acorn.shell", "cracked"), ("Acorn.surface", "wet"))
    payload["product_state"] = replace(
        payload["product_state"], policy=StoryMemoryPolicy(tombstoned_evidence_ids=("rgb-1",))
    )
    forgotten = archive.publish(**payload)
    with pytest.raises(ValueError, match="unique active ending"):
        select_film_state_evidence(plan, 2, sources(forgotten), archive)


def test_staging_scene_uses_exact_selected_revision_and_keeps_canon(tmp_path: Path) -> None:
    from duet.duetx.film_plan import FilmPlan, FilmShot
    from duet.duetx.film_staging import stage_references

    assets, archive, library = _setup(tmp_path / "archive")
    state = archive.publish(**_payload(assets, library))
    loaded = archive.load_revision(state.revision_sha256)
    root = tmp_path / "input"
    root.mkdir()
    (root / "acorn.png").write_bytes(assets.load_asset(library.references[0].media_sha256))
    plan = FilmPlan(
        "oak",
        "A new view",
        10000,
        (),
        (
            FilmShot("one", 5000, "Opening", "Hold", ("Acorn",)),
            FilmShot("two", 5000, "Another view", "Hold", ("Acorn",)),
        ),
    )
    settings = {
        "library": {"references": [{"name": "Acorn", "file": "acorn.png"}]},
        "shots": [{"world": None}, {"world": None}],
    }
    scene = {
        "source_revision_sha256": state.revision_sha256,
        "asset_sha256": state.last_frame_sha256,
    }
    names, roles = stage_references(plan, 1, settings, root, (), archive, scene)
    assert (root / names[0]).read_bytes() == loaded.last_frame_png
    assert "preceding selected scene" in roles[0]
    assert names[1] == "acorn.png"
    assert (
        archive.load_revision(state.revision_sha256).product_state.canon
        == loaded.product_state.canon
    )
    with pytest.raises(ValueError, match="does not match its recorded revision"):
        stage_references(
            plan,
            1,
            settings,
            root,
            (),
            archive,
            {**scene, "asset_sha256": library.references[0].media_sha256},
        )
