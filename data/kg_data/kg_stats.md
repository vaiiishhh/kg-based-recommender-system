# Knowledge Graph Statistics

## Entity Counts
| Entity Type | Count |
|-------------|-------|
| User        | 19,111 |
| Item        | 11,230 |
| Feature     | 20,489 |
| Brand       | 2,419 |
| Category    | 707 |
| **TOTAL**   | **53,956** |

## Relation Counts (forward relations only)
| Relation      | Head Type | Tail Type | Triple Count |
|---------------|-----------|-----------|-------------|
| purchase      | User      | Item      | 190,813 |
| mention       | User      | Feature   | 4,801,061 |
| described_by  | Item      | Feature   | 309,537 |
| produced_by   | Item      | Brand     | 11,207 |
| belong_to     | Item      | Category  | 22,416 |
| also_bought   | Item      | Item      | 35,850 |
| also_viewed   | Item      | Item      | 41,103 |

## Total Triples in KG (incl. reverse + NOOP)
| Type           | Count |
|----------------|-------|
| Forward        | 5,411,987 |
| Reverse        | 5,411,987 |
| NOOP self-loop | 53,956 |
| **TOTAL**      | **10,877,930** |

## Notes for path reasoning implementation
- NUM_ENTITIES  = 53956
- NUM_RELATIONS = 15
- NOOP relation ID = 14
- Relation IDs: {
  "purchase": 0,
  "rev_purchase": 1,
  "mention": 2,
  "rev_mention": 3,
  "described_by": 4,
  "rev_described": 5,
  "produced_by": 6,
  "rev_produced": 7,
  "belong_to": 8,
  "rev_belong": 9,
  "also_bought": 10,
  "rev_also_bought": 11,
  "also_viewed": 12,
  "rev_also_viewed": 13,
  "self_loop": 14
}