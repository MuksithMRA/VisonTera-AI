# `/agent/stats` payload reference (backend)

VisionTera AI posts detection statistics to:

- **Method:** `POST`
- **Path:** `{API_URL}/agent/stats`
- **Body:** JSON object (see below)

## Request headers (client)

| Header | Purpose |
|--------|---------|
| `Authorization` | `Bearer <access_token>` — same token used for other dashboard APIs. |
| `Content-Type` | `application/json` |
| `Accept-Language` | `en` |

---

## Root object

| Key | Type | Definition |
|-----|------|------------|
| `boxes` | `array` | One entry per **Male or Female** detection in the current frame. Detections with gender still **Person** (unknown) or any value other than Male/Female are **omitted** from this array and from `counts.counter_ids`. |
| `counts` | `object` | Aggregates for the same snapshot as `boxes` (same `date` timestamp). |

---

## `boxes[]` — each element

| Key | Type | Definition |
|-----|------|------------|
| `camera_id` | `integer` \| `string` | **Backend camera id** when the client has mapped the local camera to the API (`GET /api/cameras`). Otherwise a numeric value derived from the local `camera_id` string (digits in the id, or a stable hash fallback). **Note:** In `push_immediate` only, the client may send the raw local `camera_id` string here instead of the resolved integer. |
| `date` | `string` | ISO 8601 timestamp from the client when the snapshot was built (e.g. `2025-03-23T15:04:01.123456`). |
| `bbox_id` | `integer` | **Tracker id** for that detection (`id` in the client payload). Used for per-frame tracking; may change if the tracker reassigns ids. `-1` if missing. |
| `counter_id` | `integer` | **Stable identity for deduplication:** Re-ID **global_id** when assigned and ≥ 0; otherwise same as **`bbox_id`** (tracker id). Aligns with `counts.counter_ids` at the same index. |
| `bbox_left` | `integer` | Left edge of the bounding box in **pixels** (image coordinates). |
| `bbox_top` | `integer` | Top edge of the bounding box in **pixels**. |
| `bbox_w` | `integer` | **Width** of the box in pixels (`x2 - x1`). |
| `bbox_h` | `integer` | **Height** of the box in pixels (`y2 - y1`). |
| `gender` | `string` | `"M"` or `"F"` only (Male/Female in the client; unknown genders are not included in `boxes`). |

---

## `counts` — object

| Key | Type | Definition |
|-----|------|------------|
| `camera_id` | `integer` \| `string` | Same meaning as `boxes[].camera_id` for this snapshot (resolved backend id in the normal queue path; string possible in `push_immediate`). |
| `date` | `string` | Same ISO timestamp as in `boxes[]` for this snapshot. |
| `counter` | `integer` | **Total number of person detections** in the frame **before** gender filtering — i.e. `len(detections)` on the client, including unknown gender and persons not yet classified. May be **greater than** `len(boxes)` when some tracks are still **Person**. |
| `male_counter` | `integer` | Count of detections classified **Male** in this frame. |
| `female_counter` | `integer` | Count of detections classified **Female** in this frame. |
| `cross_counter` | `integer` | **Cumulative** count of **line-crossing events** for this camera since the stream started (or last reset): each time a tracked person’s motion segment crosses a configured counting line. **Not** a “people in frame” count. Omitted in the rare `push_immediate` payload (see below). |
| `line_counters` | `array` of `integer` | **Per-line** crossing counts: index `i` matches `counting_lines[i]` on the client. Same cumulative semantics as `cross_counter`, split by line. Omitted in `push_immediate`. |
| `counter_ids` | `array` of `integer` | Identity ids for **each** `boxes[]` row, **same order** as `boxes`: use with `counter_id` / Re-ID to deduplicate unique persons across requests. |

---

## Sample payload (typical `push_detections`)

Retail entrance camera, 1080p-style coordinates. Four people are Male/Female and appear in `boxes`; one track is still unclassified **Person** on the client, so `counter` is **5** while `boxes` has **4** rows. `cross_counter` / `line_counters` are cumulative since the stream started (two counting lines: e.g. in vs out).

```json
{
  "boxes": [
    {
      "camera_id": 14,
      "date": "2025-03-23T14:32:01.847291",
      "bbox_id": 3,
      "counter_id": 10042,
      "bbox_left": 412,
      "bbox_top": 218,
      "bbox_w": 108,
      "bbox_h": 312,
      "gender": "M"
    },
    {
      "camera_id": 14,
      "date": "2025-03-23T14:32:01.847291",
      "bbox_id": 7,
      "counter_id": 7,
      "bbox_left": 892,
      "bbox_top": 245,
      "bbox_w": 96,
      "bbox_h": 298,
      "gender": "M"
    },
    {
      "camera_id": 14,
      "date": "2025-03-23T14:32:01.847291",
      "bbox_id": 12,
      "counter_id": 10088,
      "bbox_left": 1204,
      "bbox_top": 268,
      "bbox_w": 102,
      "bbox_h": 305,
      "gender": "F"
    },
    {
      "camera_id": 14,
      "date": "2025-03-23T14:32:01.847291",
      "bbox_id": 18,
      "counter_id": 18,
      "bbox_left": 1510,
      "bbox_top": 290,
      "bbox_w": 88,
      "bbox_h": 276,
      "gender": "F"
    }
  ],
  "counts": {
    "camera_id": 14,
    "date": "2025-03-23T14:32:01.847291",
    "counter": 5,
    "male_counter": 2,
    "female_counter": 2,
    "cross_counter": 1847,
    "line_counters": [612, 1235],
    "counter_ids": [10042, 7, 10088, 18]
  }
}
```

**Sample `push_immediate`** (no line metrics; local camera id may be a string):

```json
{
  "boxes": [
    {
      "camera_id": "cam_siteA_entrance_1",
      "date": "2025-03-23T14:32:02.103005",
      "bbox_id": 4,
      "counter_id": 4,
      "bbox_left": 640,
      "bbox_top": 180,
      "bbox_w": 72,
      "bbox_h": 210,
      "gender": "F"
    }
  ],
  "counts": {
    "camera_id": "cam_siteA_entrance_1",
    "date": "2025-03-23T14:32:02.103005",
    "counter": 1,
    "male_counter": 0,
    "female_counter": 1,
    "counter_ids": [4]
  }
}
```

---

## Semantics notes

1. **Gender filter:** Only Male/Female appear in `boxes` and in `counter_ids` / per-box `counter_id`. Unknown gender does not contribute to `boxes` but **does** contribute to `counter`.
2. **Counters:** `male_counter + female_counter` can be **less than** `counter` when some tracks are unclassified.
3. **Line crossing:** `cross_counter` is the **sum over time** of all line crossings (global); `line_counters` gives the breakdown by line.
4. **Source implementation:** `app/services/api_client.py` (`push_detections`, `push_immediate`).

## `push_immediate` (edge case)

Used only for immediate pushes (bypasses queue). Payload may omit `cross_counter` and `line_counters`, and `counts.camera_id` / `boxes[].camera_id` may be the **local string** `camera_id` instead of the resolved backend integer.
