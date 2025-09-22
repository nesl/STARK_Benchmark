# [STARK: Spatial-Temporal reAsoning benchmaRK](https://arxiv.org/abs/2505.11618)

## 🔥 News
- 🔥 STARK has been accepted to [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/121374) dataset and benchmark track! Stay tuned for our camera-ready!
- 🔥 We've released both [STARK-S](https://huggingface.co/datasets/prquan/STARK_1k) (1.3k samples, for low-cost and rapid evaluation) and [STARK-L](https://huggingface.co/datasets/prquan/STARK_10k) (14k samples, for comprehensive evaluation), enabling the community to choose the benchmark size that fits their cost budget.
- We've released full dataset (STARK-L) evaluation [cost](#stark-l-evaluation-cost) for reference.

## Dataset Summary

**[STARK](https://arxiv.org/abs/2505.11618)** is a comprehensive benchmark designed to systematically evaluate large language models (LLMs) and large reasoning models (LRMs) on spatial-temporal reasoning tasks, particularly for applications in cyber-physical systems (CPS) such as robotics, autonomous vehicles, and smart city infrastructure.

- **Hierarchical Benchmark:** Tasks are structured across three levels of reasoning complexity:
  1. **State Estimation:** Field variable prediction, spatial/temporal localization, and tracking with diverse sensor modalities (range, bearing, proximity, event-based).
  2. **Reasoning Over Estimated States:** Inference of spatial, temporal, and spatiotemporal relationships using formal logic frameworks (DE-9IM for space, Allen’s interval algebra for time).
  3. **World-Knowledge-Aware Reasoning:** Context- and knowledge-rich challenges such as intent prediction, route planning, landmark reasoning, and human mobility forecasting.

- **Scale and Diversity:** Contains 25 unique tasks, over 10k challenge instances, and supports open-ended answers.

- **Sensor Modalities:** Simulates real-world data from range sensors, bearing sensors, region (proximity) sensors, and event-based (e.g., TOA) sensors, as well as real and synthetic field variable datasets (e.g., air quality, traffic, temperature).

- **Evaluation Focus:** Tasks are designed to assess both direct reasoning and the ability to generate and execute Python code. Baseline methods include classical geometric algorithms (multilateration, triangulation, Kalman filtering) and FSMs for human activity modeling.

- **Reproducibility:** All [data](https://huggingface.co/datasets/prquan/STARK_10k/edit/main/README.md), [code](https://github.com/nesl/STARK_Benchmark/), and evaluation scripts are open-sourced to encourage benchmarking and method development in spatial-temporal reasoning. Please refer to our [GitHub repo](https://github.com/nesl/STARK_Benchmark/).

## Example Use Cases

- Benchmarking LLM and LRM performance in geometric localization, trajectory tracking, spatial/temporal relationship inference, and real-world navigation tasks.
- Evaluating multi-step reasoning pipelines in simulated CPS environments.
- Assessing both direct question-answering and tool-use (Python code generation) capabilities.

## Setup
```
pip install -r requirements.txt
```

## How to use

1. Untar the data_final.tar.gz file:
```
tar -xzvf data_final.tar.gz
```
2. Rename the directory:
```
mv data_final_v5/ data
```
3.1 Put your openai token into key.txt
```
echo "YOUR_OPENAI_TOKEN" >> key.txt
```
3.2 Begin to use
```
python main.py --openai $m --dataset $t --index $i --mode $mode
```

### Explanations of arguments:
(1) --mode: Choose between {'text', 'code'}. It allows users to specify how LLMs interact with data directly through text or with Python code interpreter.

*mode* \in {'text', 'code'}

```
--mode text
```
(2) --model: Choose between ("Llama-4" "Mistral-7B" "Llama-3-8b" "deepseek-chat" "gpt-4.1" "gpt-4.5-preview" "gpt-4o" "gpt-4o-mini" "o4-mini" "o3-mini" "o3"). To use models from [together.ai](https://https://www.together.ai/pricing)(not gpt-X models from OpenAI), you will need to specify your Together.ai key in together_key.txt.
```
--model gpt-4o
```
(3) --index: The index of data sample provided for LLMs. *index* \in {1, ..., N}
```
--index 1
```
(4) --dataset: The type of spatiotemporal reasoning task in ```./task/*.txt```.
```
--dataset loc_range
```
Those tasks include:
```
tasks = ["loc_range", "loc_bearing", "loc_range_bearing", "loc_region", "loc_event_temp", "loc_event_spatio", "track_range_online", "track_bearing_online", "track_range_bearing_online", "track_region_online", "track_event_spatio_online", "track_event_temp_online", "spatial_impute", "spatiotemporal_forecast", "spatiotemporal_impute", "temporal_impute", "Point_Point_equals", "Point_Linestring_intersects", "Point_Linestring_within", "Point_Linestring_touches", "Point_Polygon_intersects", "Point_Polygon_within", "Point_Polygon_touches", "Linestring_Point_intersects", "Linestring_Point_contains", "Linestring_Point_touches", "Linestring_Linestring_equals", "Linestring_Linestring_intersects", "Linestring_Linestring_contains", "Linestring_Linestring_within", "Linestring_Linestring_crosses", "Linestring_Linestring_touches", "Linestring_Linestring_overlaps", "Linestring_Polygon_intersects", "Linestring_Polygon_within", "Linestring_Polygon_crosses", "Linestring_Polygon_touches", "Polygon_Point_intersects", "Polygon_Point_contains", "Polygon_Point_touches", "Polygon_Linestring_intersects",\

"Polygon_Linestring_contains", "Polygon_Linestring_crosses", "Polygon_Linestring_touches", "Polygon_Polygon_equals", "Polygon_Polygon_intersects", "Polygon_Polygon_contains", "Polygon_Polygon_within", "Polygon_Polygon_touches", "Polygon_Polygon_overlaps", "precedes", "is_preceded_by", "meets", "is_met_by", "overlaps_with", "is_overlapped_by", "starts", "is_started_by", "during", "contains", "finishes", "finished_by", "is_equal_to", "Linestring_Point_intersects-precedes", "Linestring_Point_intersects-meets", "Linestring_Point_intersects-overlaps_with", "Linestring_Point_intersects-starts", "Linestring_Point_intersects-during", "Linestring_Point_intersects-finishes", "Linestring_Point_intersects-is_equal_to", "Linestring_Linestring_equals-precedes", "Linestring_Linestring_equals-meets", "Linestring_Linestring_equals-overlaps_with", "Linestring_Linestring_equals-starts", "Linestring_Linestring_equals-during", "Linestring_Linestring_equals-finishes", "Linestring_Linestring_equals-is_equal_to", "Linestring_Linestring_intersects-precedes", "Linestring_Linestring_intersects-meets", "Linestring_Linestring_intersects-overlaps_with", "Linestring_Linestring_intersects-starts", "Linestring_Linestring_intersects-during", "Linestring_Linestring_intersects-finishes", "Linestring_Linestring_intersects-is_equal_to", "Linestring_Linestring_contains-precedes", "Linestring_Linestring_contains-meets", "Linestring_Linestring_contains-overlaps_with", "Linestring_Linestring_contains-starts", "Linestring_Linestring_contains-during", "Linestring_Linestring_contains-finishes", "Linestring_Linestring_contains-is_equal_to", "Linestring_Linestring_crosses-precedes", "Linestring_Linestring_crosses-meets", "Linestring_Linestring_crosses-overlaps_with", "Linestring_Linestring_crosses-starts", "Linestring_Linestring_crosses-during", "Linestring_Linestring_crosses-finishes",\

"Linestring_Linestring_crosses-is_equal_to", "Linestring_Linestring_overlaps-precedes", "Linestring_Linestring_overlaps-meets", "Linestring_Linestring_overlaps-overlaps_with", "Linestring_Linestring_overlaps-starts", "Linestring_Linestring_overlaps-during", "Linestring_Linestring_overlaps-finishes", "Linestring_Linestring_overlaps-is_equal_to", "Linestring_Polygon_intersects-precedes", "Linestring_Polygon_intersects-meets", "Linestring_Polygon_intersects-overlaps_with", "Linestring_Polygon_intersects-starts", "Linestring_Polygon_intersects-during", "Linestring_Polygon_intersects-finishes", "Linestring_Polygon_intersects-is_equal_to", "Linestring_Polygon_within-precedes", "Linestring_Polygon_within-meets", "Linestring_Polygon_within-overlaps_with", "Linestring_Polygon_within-starts", "Linestring_Polygon_within-during", "Linestring_Polygon_within-finishes", "Linestring_Polygon_within-is_equal_to", "Linestring_Polygon_crosses-precedes", "Linestring_Polygon_crosses-meets", "Linestring_Polygon_crosses-overlaps_with", "Linestring_Polygon_crosses-starts", "Linestring_Polygon_crosses-during", "Linestring_Polygon_crosses-finishes", "Linestring_Polygon_crosses-is_equal_to", "direction_questions", "intent_pred", "landmark_questions", "poi_pred", "route_planning", "travel_questions", "subroute_duration"]
```
Example usage:
```
python main.py --openai Llama-4 --dataset loc_range  --index 5 --mode text
```

## STARK-L Evaluation Cost

| Model| Cost per run | Provider|
|--|-|-|
| O3*| $759.8| OpenAI|
| O3-mini| $442.9| OpenAI|
| O4-mini| $352.6| OpenAI|
| GPT-4.1| $237.5| OpenAI|
| GPT-4o| $284.1| OpenAI|
| GPT-4o-mini| $18.3| OpenAI|
| LlaMA-4| $44.4| Together.ai  |
| LlaMA-3-8B| $9.21| Together.ai  |
| Mistral-7| $25.1| Together.ai  |
*The cost of o3 is reduced by 80% since 06/2025.


## Contact Information

If you have any questions or feedback, feel free to reach out:

- **Name:** Pengrui Quan
- **Email:** [prquan@ucla.edu](mailto:prquan@ucla.edu)

## Preprint

For more detail, refer to our [preprint](https://arxiv.org/abs/2505.11618).