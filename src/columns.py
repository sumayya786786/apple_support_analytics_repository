"""Exact Google Forms export labels and analysis variable names."""

COLUMN_MAP = {
    "timestamp": "Timestamp",
    "age_group": "Age group",
    "gender": "Gender",
    "employment_status": "Employment or student status",
    "consent": "Do you consent to take part in this survey?",
    "eligible_recent_use": "Have you used Apple retail support, Genius Bar support, Apple repair support, or in-store technical support in the last 24 months?",
    "support_type": "Which type of Apple support did you use most recently?",
    "product_type": "Which Apple product was the support related to?",
    "issue_type": "What was the main type of issue?",
    "visit_recency": "When did this support experience take place?",
    "warranty_status": "What was your warranty or cover status at the time?",
    "appointment_access": "It was easy to book or access the support I needed.",
    "waiting_acceptability": "The waiting time was acceptable.",
    "communication_quality": "Staff communicated clearly throughout the support process.",
    "staff_helpfulness": "Staff were helpful and supportive.",
    "technical_knowledge": "Staff appeared technically knowledgeable.",
    "explanation_clarity": "The explanation of the issue, repair or next steps was clear.",
    "options_clarity": "The available repair, replacement or support options were explained clearly.",
    "cost_warranty_clarity": "Any costs, warranty position or AppleCare options were explained clearly.",
    "turnaround_acceptability": "The repair or support turnaround time was acceptable.",
    "process_ease": "The overall support process was easy to follow.",
    "resolution_status": "Was your issue resolved?",
    "overall_satisfaction": "Overall, how satisfied were you with the Apple support experience?",
    "reuse_intention": "How likely would you be to use Apple support again based on this experience?",
    "recommendation_intention": "How likely would you be to recommend Apple support to someone else based on this experience?",
    "positive_comment": "What was the most positive part of your Apple support experience?",
    "negative_comment": "What was the most negative or frustrating part of your Apple support experience?",
    "improvement_comment": "Is there anything Apple could have done to improve your support experience?",
    "biggest_factor": "Overall, which one factor had the biggest impact on your satisfaction or dissatisfaction?",
}

LIKERT_COLUMNS = [
    "appointment_access",
    "waiting_acceptability",
    "communication_quality",
    "staff_helpfulness",
    "technical_knowledge",
    "explanation_clarity",
    "options_clarity",
    "cost_warranty_clarity",
    "process_ease",
]

RATING_COLUMNS = LIKERT_COLUMNS[:-1] + ["turnaround_acceptability", "process_ease"]

PRIMARY_PREDICTORS = [
    "waiting_acceptability",
    "communication_quality",
    "staff_helpfulness",
    "resolution_partial",
    "resolution_unresolved",
]

EXPLORATORY_PREDICTORS = [
    "appointment_access",
    "technical_knowledge",
    "explanation_clarity",
    "options_clarity",
    "cost_warranty_clarity",
    "process_ease",
]

TURNAROUND_MAP = {
    "1 - Strongly disagree": 1,
    "2 - Disagree": 2,
    "3 - Neither agree nor disagree": 3,
    "4 - Agree": 4,
    "5 - Strongly agree": 5,
    "Not applicable": None,
}

RESOLUTION_MAP = {
    "Yes, fully resolved": "Fully resolved",
    "Partly resolved": "Partly resolved",
    "No, not resolved": "Not resolved",
    "Not sure": "Not sure",
}

DISPLAY_LABELS = {
    "appointment_access": "Appointment access",
    "waiting_acceptability": "Waiting acceptability",
    "communication_quality": "Communication quality",
    "staff_helpfulness": "Staff helpfulness",
    "technical_knowledge": "Technical knowledge",
    "explanation_clarity": "Explanation clarity",
    "options_clarity": "Options clarity",
    "cost_warranty_clarity": "Cost/warranty clarity",
    "turnaround_acceptability": "Turnaround acceptability",
    "process_ease": "Process ease",
    "resolution_partial": "Partly resolved",
    "resolution_unresolved": "Not resolved",
}
