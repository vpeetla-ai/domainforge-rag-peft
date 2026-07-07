"""Maps Bitext intent strings to enumerated suggested_action codes."""

INTENT_TO_ACTION: dict[str, str] = {
    "track_order": "lookup_order_status",
    "change_order": "explain_return_process",
    "cancel_order": "cancel_subscription",
    "place_order": "lookup_order_status",
    "get_refund": "initiate_refund",
    "check_refund_policy": "explain_refund_policy",
    "track_refund": "track_refund_status",
    "recover_password": "reset_password",
    "registration_problems": "recover_account",
    "create_account": "recover_account",
    "delete_account": "route_to_specialist",
    "edit_account": "recover_account",
    "switch_account": "recover_account",
    "check_payment_methods": "explain_payment_methods",
    "payment_issue": "resolve_payment_issue",
    "check_invoice": "resolve_payment_issue",
    "get_invoice": "resolve_payment_issue",
    "delivery_options": "provide_tracking_info",
    "delivery_period": "provide_tracking_info",
    "change_shipping_address": "provide_tracking_info",
    "set_up_shipping_address": "provide_tracking_info",
    "contact_customer_service": "explain_working_hours",
    "contact_human_agent": "escalate_to_supervisor",
    "complaint": "escalate_to_supervisor",
    "review": "no_action_required",
    "newsletter_subscription": "no_action_required",
    "check_cancellation_fee": "explain_cancellation_options",
}


def intent_to_action(intent: str) -> str:
    return INTENT_TO_ACTION.get(intent, "route_to_specialist")
