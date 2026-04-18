# Slot Overbooking and Standby Patient Management Guideline

## Purpose
This guideline provides a framework for managing appointment overbooking to maximize clinic utilization while maintaining patient care quality. Overbooking compensates for predicted no-shows by strategically scheduling additional patients.

## When to Activate Overbooking
- When the AI prediction system identifies a significant number of high-risk no-show appointments in a given time slot or day.
- Overbooking should only be applied to slots where the predicted no-show probability exceeds 55%.
- Maximum overbooking rate: 10–15% of total scheduled appointments per session.

## Standby Patient List
- Maintain a rolling standby list of patients who can attend on short notice.
- Priority for standby: Patients whose own appointments are far in the future, patients requesting earlier appointments, and follow-up patients.
- Standby patients should be contacted 4 hours before the slot becomes available.
- Standby patients must confirm within 1 hour of being contacted.

## Overbooking Capacity Rules
- **Small clinic (≤5 providers)**: Maximum 1 overbooking per provider per half-day session.
- **Large clinic (>5 providers)**: Maximum 2 overbookings per provider per half-day session.
- Never overbook specialty consultations or procedures that require specific preparation.
- Emergency/urgent slots must never be overbooked.

## Staff Workflow
1. Review AI-generated risk report each morning.
2. Identify high-risk slots eligible for overbooking.
3. Contact standby patients for eligible slots.
4. Update scheduling system to reflect overbooking status.
5. Monitor day-of attendance and adjust as confirmations come in.

## Patient Experience Considerations
- If both the original patient and standby patient arrive, accommodate both without exceeding 15-minute wait time increase.
- Apologize for any inconvenience and document the situation.
- Review overbooking decisions weekly to calibrate the model threshold.

## Risk Mitigation
- Track overbooking outcomes: percentage of times both patients show up.
- If double-attendance exceeds 20% of overbooked slots, increase the no-show probability threshold for overbooking activation.
- Monthly audit of overbooking impact on patient satisfaction scores.
