"""Local browser interface; model execution stays in the inference worker."""

from __future__ import annotations

import math
from pathlib import Path

import gradio as gr

from .catalog import (
    CAPTION_PROMPT,
    COMPLETE_PROMPT,
    COMPONENTS,
    DETAIL_PROMPT,
    MISSING_PROMPT,
    PANOPTIC_PROMPT,
    SEGMENT_PROMPT,
    SUMMARY_PROMPT,
    category_prompt,
    referring_prompt,
)
from .images import load_image, parse_components, presence_rows, render_prediction


def _answer(response):
    answer = response.get("result", {}).get("answer", "")
    return answer if isinstance(answer, str) else str(answer)


def _metrics(responses):
    metrics = [response.get("metrics", {}) for response in responses]
    elapsed = sum(float(item.get("seconds", 0)) for item in metrics)
    allocated = max((float(item.get("peak_allocated_mib", 0)) for item in metrics), default=0) / 1024
    reserved = max((float(item.get("peak_reserved_mib", 0)) for item in metrics), default=0) / 1024
    return f"Inference: {elapsed:.2f} s · Peak GPU memory: {allocated:.2f} GiB allocated / {reserved:.2f} GiB reserved"


def _differences(left, right):
    if left is None or right is None:
        return []
    original, variant = dict(presence_rows(left)), dict(presence_rows(right))
    rows = []
    for name in COMPONENTS:
        before, after = original.get(name), variant.get(name)
        numeric = isinstance(before, (int, float)) and isinstance(after, (int, float))
        delta = round(after - before, 4) if numeric else "Unavailable"
        if isinstance(delta, float) and not math.isfinite(delta):
            delta = "Unavailable"
        rows.append([name, delta])
    return rows


def build_app(client, catalog=None):
    """Build the two-tab UI without loading weights or starting a web server."""
    choices = catalog.choices if catalog else []
    initial_source = catalog.sources[0] if catalog and catalog.sources else None
    initial_variants = catalog.variants(initial_source) if initial_source else []
    initial_path = str(catalog.path(initial_source)) if initial_source else None
    initial_questions = catalog.questions(initial_source) if initial_source else []
    initial_targets = [item for item in initial_variants if catalog.is_counterfactual(item[1])]
    initial_target = initial_targets[0][1] if initial_targets else None
    thumbnails = []
    for label, key in choices:
        thumbnail = load_image(catalog.path(key))
        thumbnail.thumbnail((180, 140))
        thumbnails.append((thumbnail, label))

    local_event = {"queue": False, "api_visibility": "private"}
    gpu_event = {"concurrency_id": "falcon_gpu", "concurrency_limit": 1,
                 "api_visibility": "private"}

    with gr.Blocks(title="FALCON", analytics_enabled=False, delete_cache=(3600, 3600)) as app:
        gr.Markdown("# FALCON\nExplore X-ray images and component predictions.")
        status = gr.Textbox(label="Model", value="Loading model…", interactive=False)
        gr.Markdown("Research demo. Component completeness is not a danger assessment.")

        with gr.Tab("Explore"):
            image_state = gr.State({"path": initial_path, "counterfactual": False, "revision": 0})
            if choices:
                gallery = gr.Gallery(thumbnails, label="Samples", columns=6, rows=2,
                                     height=270, interactive=False, format="png",
                                     buttons=[], allow_preview=False)
            with gr.Row(visible=bool(choices)):
                source = gr.Dropdown(choices, value=initial_source, label="Original sample")
                variant = gr.Dropdown(initial_variants, value=initial_source,
                                      label="Image variant")
            with gr.Row():
                with gr.Column():
                    upload = gr.File(label="Upload a PNG or JPEG (up to 20 MB / 16 MP)",
                                     file_types=[".png", ".jpg", ".jpeg"], type="filepath")
                    image = gr.Image(value=load_image(initial_path) if initial_path else None,
                                     label="Input image", interactive=False, type="filepath",
                                     format="png", buttons=["fullscreen", "download"])
                    origin = gr.Textbox(value=f"Sample: {Path(initial_path).name}" if initial_path else "",
                                        label="Selected image", interactive=False)
                with gr.Column():
                    mode = gr.Radio(["Describe", "Ask", "Ground & segment", "Components"],
                                    value="Describe", label="Task")
                    with gr.Group() as describe_controls:
                        detail = gr.Radio(["Short", "Detailed"], value="Short", label="Description")
                    with gr.Group(visible=False) as ask_controls:
                        preset = gr.Dropdown(initial_questions, value=None, label="Suggested question")
                        question = gr.Textbox(label="Question", lines=2,
                                              placeholder="Ask about this image.")
                    with gr.Group(visible=False) as ground_controls:
                        grounding = gr.Radio(["All components", "Category", "Referring expression", "Panoptic"],
                                             value="All components", label="Output")
                        category = gr.Dropdown(list(COMPONENTS), value=COMPONENTS[0],
                                               label="Component", visible=False)
                        expression = gr.Textbox(label="Referring expression", visible=False,
                                                placeholder="The battery on the right")
                    with gr.Accordion("Generation settings", open=False):
                        tokens = gr.Dropdown([64, 128, 256], value=128, label="Maximum new tokens")
                    run = gr.Button("Run FALCON", variant="primary", interactive=False)
            with gr.Row():
                overlay = gr.Image(label="Predicted segmentation", interactive=False, format="png",
                                   buttons=["fullscreen", "download"])
                with gr.Column():
                    answer = gr.Textbox(label="Model answer", lines=5, interactive=False)
                    scores = gr.Dataframe(headers=["Component", "Presence score"],
                                          datatype=["str", "str"], value=[], interactive=False)
                    legend = gr.Textbox(label="Prediction details", interactive=False, lines=2)
                    timing = gr.Textbox(label="Runtime", interactive=False)
            explore_outputs = [overlay, answer, scores, legend, timing]

        with gr.Tab("Compare"):
            compare_state = gr.State({"revision": 0})
            if not choices:
                gr.Markdown("Provide `--dataset` at startup to compare existing counterfactuals.")
            with gr.Row():
                compare_source = gr.Dropdown(choices, value=initial_source, label="Original",
                                             interactive=bool(choices))
                compare_target = gr.Dropdown(initial_targets, value=initial_target,
                                             label="Counterfactual", interactive=bool(initial_targets))
                compare_tokens = gr.Dropdown([64, 128, 256], value=128, label="Maximum new tokens")
            with gr.Row():
                left_image = gr.Image(value=load_image(initial_path) if initial_path else None,
                                      label="Original", interactive=False, format="png",
                                      buttons=["fullscreen", "download"])
                right_image = gr.Image(value=load_image(catalog.path(initial_target)) if initial_target else None,
                                       label="Counterfactual", interactive=False, format="png",
                                       buttons=["fullscreen", "download"])
            compare_run = gr.Button("Compare with FALCON", variant="primary", interactive=False)
            with gr.Row():
                with gr.Column():
                    left_answer = gr.Textbox(label="Original prediction", lines=4, interactive=False)
                    left_scores = gr.Dataframe(headers=["Component", "Presence score"],
                                               datatype=["str", "str"], value=[], interactive=False)
                with gr.Column():
                    right_answer = gr.Textbox(label="Counterfactual prediction", lines=4, interactive=False)
                    right_scores = gr.Dataframe(headers=["Component", "Presence score"],
                                                datatype=["str", "str"], value=[], interactive=False)
            differences = gr.Dataframe(headers=["Component", "Counterfactual − original"],
                                       datatype=["str", "str"], value=[], interactive=False)
            compare_timing = gr.Textbox(label="Runtime", interactive=False)
            compare_outputs = [left_answer, left_scores, right_answer, right_scores,
                               differences, compare_timing]

        def clear_explore(state):
            state["revision"] += 1
            return None, "", [], "", ""

        def clear_compare(state):
            state["revision"] += 1
            return "", [], "", [], [], ""

        def predict(state, task, description, prompt, segment_mode, component, reference, limit):
            revision = state["revision"]
            if not state["path"]:
                return None, "Select a sample or upload an image first.", [], "", ""
            try:
                current = load_image(state["path"])
                responses = []
                result_mode = "text"
                if task == "Describe":
                    prompt = (DETAIL_PROMPT if description == "Detailed" else
                              CAPTION_PROMPT if state.get("counterfactual") else SUMMARY_PROMPT)
                elif task == "Ask":
                    if not prompt or not prompt.strip():
                        raise ValueError("Enter a question first.")
                elif task == "Ground & segment":
                    result_mode = "panoptic" if segment_mode == "Panoptic" else "segmentation"
                    prompt = PANOPTIC_PROMPT if result_mode == "panoptic" else SEGMENT_PROMPT
                    if segment_mode == "Category":
                        prompt = category_prompt(component)
                    elif segment_mode == "Referring expression":
                        prompt = referring_prompt(reference)
                elif task == "Components":
                    missing = client.predict(current, MISSING_PROMPT, max_new_tokens=int(limit))
                    responses.append(missing)
                    complete = client.predict(current, COMPLETE_PROMPT, max_new_tokens=int(limit))
                    responses.append(complete)
                    result = (None, parse_components(missing, complete), presence_rows(missing),
                              "Presence scores and generated answers are separate predictions.", _metrics(responses))
                else:
                    raise ValueError("Unknown task.")
                if task != "Components":
                    response = client.predict(current, prompt.strip(), mode=result_mode,
                                              max_new_tokens=int(limit))
                    responses.append(response)
                    rendered, details = render_prediction(current, response, client.workspace)
                    result = rendered, _answer(response), presence_rows(response), details, _metrics(responses)
            except (RuntimeError, ValueError, OSError) as exc:
                result = None, f"Unable to run prediction: {exc}", [], "", ""
            return result if revision == state["revision"] else (gr.skip(),) * len(explore_outputs)

        def compare(source_key, target_key, limit, state):
            revision = state["revision"]
            if not catalog or source_key not in catalog.sources:
                return "Select an original sample.", [], "", [], [], ""
            targets = {key for _, key in catalog.variants(source_key) if catalog.is_counterfactual(key)}
            if target_key not in targets:
                return "", [], "Select a counterfactual for this original.", [], [], ""
            responses, outputs = [], []
            for key in (source_key, target_key):
                try:
                    response = client.predict(load_image(catalog.path(key)), CAPTION_PROMPT,
                                              max_new_tokens=int(limit))
                    outputs.extend([_answer(response), presence_rows(response)])
                except (RuntimeError, ValueError, OSError) as exc:
                    response = None
                    outputs.extend([f"Unable to run prediction: {exc}", []])
                responses.append(response)
            outputs.extend([_differences(*responses), _metrics([item for item in responses if item])])
            return outputs if revision == state["revision"] else (gr.skip(),) * len(compare_outputs)

        run_event = run.click(predict, [image_state, mode, detail, question, grounding,
                                        category, expression, tokens], explore_outputs, **gpu_event)
        compare_event = compare_run.click(compare, [compare_source, compare_target, compare_tokens,
                                                    compare_state], compare_outputs, **gpu_event)

        def uploaded(path, state):
            cleared = clear_explore(state)
            state["path"] = None
            state["counterfactual"] = False
            if not path:
                return None, "", gr.Dropdown(choices=[], value=None), "", *cleared
            try:
                loaded = load_image(path)
                state["path"] = str(path)
                return loaded, f"Upload: {Path(path).name}", gr.Dropdown(choices=[], value=None), "", *cleared
            except (ValueError, OSError) as exc:
                return None, f"Invalid upload: {exc}", gr.Dropdown(choices=[], value=None), "", *cleared

        upload.upload(uploaded, [upload, image_state], [image, origin, preset, question, *explore_outputs],
                      cancels=[run_event], **local_event)
        upload.clear(lambda state: uploaded(None, state), image_state,
                     [image, origin, preset, question, *explore_outputs],
                     cancels=[run_event], **local_event)

        def sample_changed(key, state):
            cleared = clear_explore(state)
            state["path"] = None
            state["counterfactual"] = False
            if not key:
                return None, "", gr.Dropdown(choices=[], value=None), "", *cleared, None
            loaded = load_image(catalog.path(key))
            state["path"] = str(catalog.path(key))
            state["counterfactual"] = catalog.is_counterfactual(key)
            return loaded, f"Sample: {catalog.path(key).name}", gr.Dropdown(choices=catalog.questions(key), value=None), "", *cleared, None

        if choices:
            def select_source(key):
                return gr.Dropdown(choices=catalog.variants(key), value=key)

            def gallery_selected(state, event: gr.SelectData):
                key = choices[int(event.index)][1]
                return key, select_source(key), *sample_changed(key, state)

            gallery.select(gallery_selected, image_state,
                           [source, variant, image, origin, preset, question, *explore_outputs, upload],
                           cancels=[run_event], **local_event)
            source.change(select_source, source, variant, **local_event)
            variant.change(sample_changed, [variant, image_state],
                           [image, origin, preset, question, *explore_outputs, upload],
                           cancels=[run_event], **local_event)

        def task_changed(task, state):
            return (gr.Group(visible=task == "Describe"), gr.Group(visible=task == "Ask"),
                    gr.Group(visible=task == "Ground & segment"), *clear_explore(state))

        mode.change(task_changed, [mode, image_state],
                    [describe_controls, ask_controls, ground_controls, *explore_outputs],
                    cancels=[run_event], **local_event)
        grounding.change(lambda kind: (gr.Dropdown(visible=kind == "Category"),
                                       gr.Textbox(visible=kind == "Referring expression")),
                         grounding, [category, expression], **local_event)
        preset.change(lambda value: value or "", preset, question, **local_event)
        for control in (detail, question, grounding, category, expression, tokens):
            control.change(clear_explore, image_state, explore_outputs, cancels=[run_event], **local_event)

        def compare_source_changed(key, state):
            cleared = clear_compare(state)
            targets = [item for item in catalog.variants(key) if catalog.is_counterfactual(item[1])]
            target = targets[0][1] if targets else None
            return (load_image(catalog.path(key)),
                    gr.Dropdown(choices=targets, value=target, interactive=bool(targets)), *cleared)

        def compare_target_changed(key, state):
            return (load_image(catalog.path(key)) if key else None, *clear_compare(state))

        if choices:
            compare_source.change(compare_source_changed, [compare_source, compare_state],
                                  [left_image, compare_target, *compare_outputs],
                                  cancels=[compare_event], **local_event)
            compare_target.change(compare_target_changed, [compare_target, compare_state],
                                  [right_image, *compare_outputs], cancels=[compare_event], **local_event)
        compare_tokens.change(clear_compare, compare_state, compare_outputs,
                              cancels=[compare_event], **local_event)

        def poll_status():
            current = client.status()
            ready = current["state"] in ("ready", "running")
            return (f"{current['state'].capitalize()}: {current.get('message', '')}",
                    gr.Button(interactive=ready), gr.Button(interactive=ready and bool(choices)))

        app.load(poll_status, outputs=[status, run, compare_run], **local_event)
        gr.Timer(2).tick(poll_status, outputs=[status, run, compare_run], **local_event)
    return app.queue(max_size=8, default_concurrency_limit=1)
