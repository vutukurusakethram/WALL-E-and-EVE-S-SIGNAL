# Mission report: mission_02_walls_noise_decoys

**Outcome:** CONFIRMED  |  **Ended by:** declared

| Item | Value |
|---|---|
| Inferred EVE location | (18, 4) |
| Confidence | 100.0% (probability within detection radius of that cell) |
| 90% credible region | 1 cells |
| EVE status | active |
| Steps / scans / recharges | 105 / 6 / 1 |
| Battery left | 39 / 80 |
| Time left | 281 |

## Evidence found

- Initial signal: coords [15.0, 14.0], strength 0.4
- Strongest scan readings: 0.84 at (18, 5), 0.37 at (15, 14), 0.30 at (15, 9)

| Clue | Found at | Points to | Distance to inferred EVE | Verdict |
|---|---|---|---|---|
| tire tracks | (15, 6) | (19.0, 11.0) | 7.07 | conflicting (likely decoy) |
| weak beacon ping | (13, 8) | (19.0, 3.86) | 1.01 | consistent |
| tire tracks | (19, 10) | (12.0, 19.0) | 16.16 | conflicting (likely decoy) |
| scorch marks | (19, 3) | (12.0, 18.0) | 15.23 | conflicting (likely decoy) |

## Recommended next step

Proceed to EVE at (18, 4): 1 moves (1 battery). Battery 39 covers the trip there and back to the charger with margin.

## Evaluation only (ground truth, never shown to the agent)

True EVE position (18, 4) (active); error 0.0 cells; correct: **True**; decoy clues in this world: 4.
