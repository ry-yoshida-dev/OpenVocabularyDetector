# florence2

## Overview

Florence-2 through `transformers.Florence2ForConditionalGeneration` (`DetectorBackend.FLORENCE2`), using its
`<OPEN_VOCABULARY_DETECTION>` task.

Florence-2 generates the boxes of one phrase as location tokens, so every text query is generated separately over the
mini-batch (beam search) and its boxes are assigned to that query without phrase matching. Location tokens are read
at bin centers without rounding to whole pixels; polygon answers are bounded by boxes, and answers mixing boxes and
polygons are read in generation order.

Inference cost grows linearly with the number of text queries: each one is a separate beam search
(`BEAM_COUNT` = 3 beams, up to `MAX_NEW_TOKENS` = 1024 tokens). Images are preprocessed only once per mini-batch
and shared by every query. Keep prompts short, or prefer a detector that scores
every query in one pass when a prompt has many queries.

The model gives no score, so every box has confidence `1.0`: the confidence threshold removes nothing, NMS keeps the
first of overlapping boxes in query order, and a phrase absent from the image usually still gets a box.

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `Florence2Detector`. |
| [location_parser.py](./location_parser.py) | `LocationParser`: reads generated location tokens into normalized XYXY boxes. |
| [token_generator.py](./token_generator.py) | `TokenGenerator`: protocol through which `generate` is called with static types. |
