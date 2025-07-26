import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import os
import json
from typing import TypedDict, Dict, List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from uuid import uuid4
from datetime import datetime

# Setting up page configuration
st.set_page_config(page_title="Aarong Clothing App", layout="wide", initial_sidebar_state="expanded")

# Customer Support Chatbot Setup
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str
    order_id: str
    needs_escalation: bool
    session_id: str
    conversation_history: List[Dict[str, str]]

def create_dataset():
    dataset = {
        "queries": [
            {"query": "What services do you offer?", "category": "General", "expected_response": "We offer a wide range of Aarong clothing, including traditional and modern apparel."},
            {"query": "How can I place an order?", "category": "General", "expected_response": "Visit our website or app, select your clothing items, and follow the checkout process."},
            {"query": "What are your operating hours?", "category": "General", "expected_response": "Our online store is available 24/7. Physical stores are open 10 AM to 8 PM."},
            {"query": "How do I contact customer support?", "category": "General", "expected_response": "Use this chatbot, email support@aarong.com, or call +880-123-456-7890."},
            {"query": "Where is your service available?", "category": "General", "expected_response": "Check delivery areas on our website's 'Shipping Info' section."},
            {"query": "What are your terms and conditions?", "category": "General", "expected_response": "View our terms on the 'Terms' page of our website."},
            {"query": "Why was my order delivered late?", "category": "Complaints", "expected_response": "Check tracking or escalate if no order ID."},
            {"query": "I received the wrong item.", "category": "Complaints", "expected_response": "Initiate a return or replacement."},
            {"query": "The product was damaged.", "category": "Complaints", "expected_response": "Initiate a return or refund."},
            {"query": "The staff was rude.", "category": "Complaints", "expected_response": "File a complaint via website."},
            {"query": "Why was my order canceled?", "category": "Complaints", "expected_response": "Escalate for investigation."},
            {"query": "Why is delivery unavailable in my area?", "category": "Complaints", "expected_response": "Check service availability or escalate."},
            {"query": "How do I request a refund?", "category": "Refunds", "expected_response": "Submit a refund request via website."},
            {"query": "When will I receive my refund?", "category": "Refunds", "expected_response": "Provide refund processing details."},
            {"query": "Why haven’t I received my refund?", "category": "Refunds", "expected_response": "Check refund status or escalate."},
            {"query": "Can I get a refund for a defective product?", "category": "Refunds", "expected_response": "Initiate refund for poor quality."},
            {"query": "Are delivery fees refundable?", "category": "Refunds", "expected_response": "Explain refund policy for fees."},
            {"query": "What are the refund conditions?", "category": "Refunds", "expected_response": "Provide refund conditions."},
            {"query": "Where is my order?", "category": "Delivery", "expected_response": "Track order on website."},
            {"query": "Why is my delivery delayed?", "category": "Delivery", "expected_response": "Check tracking or offer discount."},
            {"query": "The rider couldn’t find my address.", "category": "Delivery", "expected_response": "Update address or contact rider."},
            {"query": "Can I change my delivery address?", "category": "Delivery", "expected_response": "Guide to change address."},
            {"query": "What if I’m not available for delivery?", "category": "Delivery", "expected_response": "Explain delivery retry policy."},
            {"query": "Delivery marked completed but not received.", "category": "Delivery", "expected_response": "Escalate for investigation."},
            {"query": "Why was I charged incorrectly?", "category": "Payments", "expected_response": "Verify charge or escalate."},
            {"query": "How do I pay using bKash?", "category": "Payments", "expected_response": "Guide on payment methods."},
            {"query": "Why is my payment method declined?", "category": "Payments", "expected_response": "Troubleshoot payment issue."},
            {"query": "Can I use multiple payment methods?", "category": "Payments", "expected_response": "Explain payment method policy."},
            {"query": "Why was I charged extra fees?", "category": "Payments", "expected_response": "Explain fee structure."},
            {"query": "How do I refund an overcharge?", "category": "Payments", "expected_response": "Guide to refund for overcharge."},
            {"query": "How do I log into my account?", "category": "Account", "expected_response": "Guide to account creation/login."},
            {"query": "How do I reset my password?", "category": "Account", "expected_response": "Reset password via website."},
            {"query": "Why is my account locked?", "category": "Account", "expected_response": "Explain account status or escalate."},
            {"query": "How do I update my account details?", "category": "Account", "expected_response": "Update details in profile."},
            {"query": "Why can’t I access some features?", "category": "Account", "expected_response": "Troubleshoot account access."},
            {"query": "How do I delete my account?", "category": "Account", "expected_response": "Guide to account deletion."},
            {"query": "How do I apply a promo code?", "category": "Promotions", "expected_response": "Guide to apply promo code."},
            {"query": "Why isn’t my promo code working?", "category": "Promotions", "expected_response": "Troubleshoot promo code issue."},
            {"query": "What are the promo terms?", "category": "Promotions", "expected_response": "Provide promotion terms."},
            {"query": "Can I use multiple vouchers?", "category": "Promotions", "expected_response": "Explain voucher stacking policy."},
            {"query": "Am I eligible for a discount?", "category": "Promotions", "expected_response": "Check promotion eligibility."},
            {"query": "Why was my discount not applied?", "category": "Promotions", "expected_response": "Investigate cashback issue."},
            {"query": "I need a live agent.", "category": "Escalation", "expected_response": "Escalate to live agent."},
            {"query": "The chatbot didn’t help.", "category": "Escalation", "expected_response": "Escalate to live agent."},
            {"query": "How long for a representative to respond?", "category": "Escalation", "expected_response": "Provide response time estimate."}
        ],
        "agents": [
            {"name": "Agent John", "contact_number": "+880-1234-567890", "specialty": "Complaints, Refunds"},
            {"name": "Agent Sarah", "contact_number": "+880-9876-543210", "specialty": "Delivery, Payments"},
            {"name": "Agent Ayesha", "contact_number": "+880-5555-123456", "specialty": "Account, Promotions, General, Escalation"}
        ]
    }
    with open("support_dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)
    return dataset

def load_dataset():
    if not os.path.exists("support_dataset.json"):
        return create_dataset()
    with open("support_dataset.json", "r") as f:
        return json.load(f)

def handle_default_query(query: str) -> tuple[str, bool]:
    query_lower = query.lower().strip()
    if query_lower in ["hi", "hello", "hey"]:
        return "Hello! How can I assist you today?", False
    elif query_lower in ["bye", "goodbye", "thank you", "thanks"]:
        return "Thank you for reaching out! Have a great day!", False
    return None, False

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)


# llm = ChatGroq(
#     temperature=0,
#     groq_api_key=st.secrets["GROQ_API_KEY"],
#     model_name="llama-3.3-70b-versatile"
# )

def categorize(state: State) -> State:
    default_response, needs_escalation = handle_default_query(state["query"])
    if default_response:
        return {
            "response": default_response,
            "category": "Default",
            "needs_escalation": needs_escalation,
            "sentiment": "Neutral",
            "order_id": "None"
        }
    dataset = load_dataset()
    prompt = ChatPromptTemplate.from_template(
        "Given the following dataset of queries:\n{dataset}\n"
        "Categorize the following customer query into one of these categories: Complaints, Refunds, Delivery, Payments, Account, Promotions, General, Escalation, Default. "
        "Query: {query}"
    )
    chain = prompt | llm
    dataset_str = json.dumps(dataset["queries"], indent=2)
    category = chain.invoke({"query": state["query"], "dataset": dataset_str}).content
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    if state["category"] == "Default":
        return {"sentiment": "Neutral"}
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    chain = prompt | llm
    sentiment = chain.invoke({"query": state["query"]}).content
    return {"sentiment": sentiment}

def extract_order_id(state: State) -> State:
    if state["category"] == "Default":
        return {"order_id": "None"}
    prompt = ChatPromptTemplate.from_template(
        "If the following query contains an order ID, extract and return it. If no order ID is present, return 'None'. Query: {query}"
    )
    chain = prompt | llm
    order_id = chain.invoke({"query": state["query"]}).content
    return {"order_id": order_id}

def update_conversation_history(state: State) -> State:
    if not state.get("response"):
        return state
    history = state.get("conversation_history", [])
    history.append({"query": state["query"], "response": state["response"], "timestamp": str(datetime.now())})
    return {"conversation_history": history}

def handle_complaints(state: State) -> State:
    query_lower = state["query"].lower()
    order_id = state.get("order_id", "None")
    if "late" in query_lower or "not delivered" in query_lower:
        response = (
            f"We're sorry for the issue with your order (Order ID: {order_id}). "
            "Please check the tracking details on our website using your order ID. "
            "As a gesture of goodwill, we've applied a 10% discount code (DELAY10) to your next order."
        )
        needs_escalation = state["sentiment"] == "Negative" and order_id == "None"
    elif "wrong item" in query_lower or "incomplete" in query_lower:
        response = (
            f"We apologize for receiving the wrong or incomplete order (Order ID: {order_id}). "
            "Please initiate a return or replacement via the 'Returns' section on our website with your order ID."
        )
        needs_escalation = state["sentiment"] == "Negative" and order_id == "None"
    elif "damaged" in query_lower or "stale" in query_lower or "poor quality" in query_lower:
        response = (
            f"We apologize for the poor quality of your order (Order ID: {order_id}). "
            "Please report this via the 'Returns' section on our website to initiate a refund or replacement."
        )
        needs_escalation = state["sentiment"] == "Negative" and order_id == "None"
    elif "rude" in query_lower or "unprofessional" in query_lower:
        response = (
            f"We're sorry for your experience with our service provider. "
            "Please file a complaint via the 'Support' section on our website, and we'll investigate promptly."
        )
        needs_escalation = True
    elif "canceled without notification" in query_lower:
        response = (
            f"We apologize for the cancellation of your order (Order ID: {order_id}) without notification. "
            "This issue requires further investigation."
        )
        needs_escalation = True
    elif "service unavailable" in query_lower:
        response = (
            f"We're sorry, our service is currently unavailable in your area. "
            "Please check the 'Service Areas' section on our website for updates on expansion."
        )
        needs_escalation = state["sentiment"] == "Negative"
    else:
        response = (
            f"We apologize for the issue with your order (Order ID: {order_id}). "
            "Please provide more details about your complaint to assist you better."
        )
        needs_escalation = True
    return {"response": response, "needs_escalation": needs_escalation}

def handle_refunds(state: State) -> State:
    query_lower = state["query"].lower()
    order_id = state.get("order_id", "None")
    if "canceled" in query_lower or "undelivered" in query_lower:
        response = (
            f"To request a refund for your canceled or undelivered order (Order ID: {order_id}), visit the 'Refund' section on our website and submit a request with your order ID. "
            "Refunds are processed within 3-5 business days."
        )
        needs_escalation = state["sentiment"] == "Negative" and order_id == "None"
    elif "when will i receive my refund" in query_lower or "how will it be processed" in query_lower:
        response = (
            f"Refunds for Order ID: {order_id} are processed within 3-5 business days via the original payment method (e.g., bKash, bank transfer) or as a voucher, depending on your preference."
        )
        needs_escalation = False
    elif "haven’t received my refund" in query_lower:
        response = (
            f"We're sorry for the delay in processing your refund (Order ID: {order_id}). "
            "Please check the 'Refund Status' section on our website or provide more details for assistance."
        )
        needs_escalation = state["sentiment"] == "Negative"
    elif "poor quality" in query_lower:
        response = (
            f"For a refund due to poor quality (Order ID: {order_id}), please submit a request via the 'Refund' section on our website with details and photos of the issue."
        )
        needs_escalation = state["sentiment"] == "Negative" and order_id == "None"
    elif "delivery fees" in query_lower or "tips refundable" in query_lower:
        response = (
            f"Delivery fees and tips are refundable only if the order was not delivered or canceled before dispatch (Order ID: {order_id}). "
            "Please check our refund policy on the website."
        )
        needs_escalation = False
    elif "conditions" in query_lower:
        response = (
            f"Refunds are available for canceled, undelivered, or poor-quality orders (Order ID: {order_id}). "
            "Please review our refund policy in the 'Support' section on our website."
        )
        needs_escalation = False
    else:
        response = (
            f"To request a refund (Order ID: {order_id}), visit the 'Refund' section on our website and follow the instructions."
        )
        needs_escalation = state["sentiment"] == "Negative"
    return {"response": response, "needs_escalation": needs_escalation}

def handle_delivery(state: State) -> State:
    query_lower = state["query"].lower()
    order_id = state.get("order_id", "None")
    if "where is my order" in query_lower or "track" in query_lower:
        response = (
            f"To track your order (Order ID: {order_id}), visit the 'Track Order' section on our website or app for real-time updates."
        )
        needs_escalation = state["sentiment"] == "Negative" and order_id == "None"
    elif "delayed" in query_lower or "when will it arrive" in query_lower:
        response = (
            f"We apologize for the delay in your order (Order ID: {order_id}). "
            "Please check the tracking details on our website. "
            "As a gesture of goodwill, use code DELAY10 for a 10% discount on your next order."
        )
        needs_escalation = state["sentiment"] == "Negative" and order_id == "None"
    elif "couldn’t find my address" in query_lower:
        response = (
            f"We're sorry the rider couldn't find your address for Order ID: {order_id}. "
            "Please verify your address in the 'Profile' section or contact the rider via the app."
        )
        needs_escalation = state["sentiment"] == "Negative"
    elif "change my delivery address" in query_lower:
        response = (
            f"To change your delivery address for Order ID: {order_id}, visit the 'Order Details' section on our website or app before dispatch. "
            "Contact support if the order is already in transit."
        )
        needs_escalation = False
    elif "not available to receive" in query_lower:
        response = (
            f"If you're unavailable to receive your delivery (Order ID: {order_id}), our rider will attempt redelivery. "
            "Check our delivery policy on the website for details."
        )
        needs_escalation = False
    elif "marked as completed" in query_lower:
        response = (
            f"We're sorry your order (Order ID: {order_id}) was marked as completed but not received. "
            "This issue requires further investigation."
        )
        needs_escalation = True
    else:
        response = (
            f"For delivery issues with Order ID: {order_id}, please check the 'Track Order' section on our website or provide more details."
        )
        needs_escalation = state["sentiment"] == "Negative"
    return {"response": response, "needs_escalation": needs_escalation}

def handle_payments(state: State) -> State:
    query_lower = state["query"].lower()
    order_id = state.get("order_id", "None")
    if "incorrectly" in query_lower or "without permission" in query_lower:
        response = (
            f"We're sorry for the incorrect charge on Order ID: {order_id}. "
            "Please verify the transaction details in your account or provide more information for investigation."
        )
        needs_escalation = True
    elif "how do i pay" in query_lower or "bkash" in query_lower or "credit card" in query_lower or "cash-on-delivery" in query_lower:
        response = (
            f"You can pay using bKash, credit/debit cards, or cash-on-delivery. "
            "Select your preferred method in the 'Payments' section during checkout for Order ID: {order_id}."
        )
        needs_escalation = False
    elif "not working" in query_lower or "declined" in query_lower:
        response = (
            f"If your payment method for Order ID: {order_id} is not working, ensure sufficient funds and correct details. "
            "Try another method or contact your bank."
        )
        needs_escalation = state["sentiment"] == "Negative"
    elif "multiple payment methods" in query_lower:
        response = (
            f"Currently, we do not support multiple payment methods for a single order (Order ID: {order_id}). "
            "Please select one method during checkout."
        )
        needs_escalation = False
    elif "additional fees" in query_lower:
        response = (
            f"Additional fees (e.g., delivery, platform, VAT) for Order ID: {order_id} are listed at checkout. "
            "Review our fee structure in the 'Support' section."
        )
        needs_escalation = False
    elif "refund for incorrect payment" in query_lower or "overcharge" in query_lower:
        response = (
            f"To request a refund for an incorrect payment or overcharge (Order ID: {order_id}), submit a request in the 'Refund' section on our website."
        )
        needs_escalation = state["sentiment"] == "Negative"
    else:
        response = (
            f"For payment issues with Order ID: {order_id}, visit the 'Payments' section on our website or provide more details."
        )
        needs_escalation = state["sentiment"] == "Negative"
    return {"response": response, "needs_escalation": needs_escalation}

def handle_account(state: State) -> State:
    query_lower = state["query"].lower()
    if "create" in query_lower or "log in" in query_lower:
        response = (
            "To create or log into your account, visit the 'Sign Up' or 'Login' page on our website or app and follow the instructions."
        )
        needs_escalation = False
    elif "forgot my password" in query_lower or "reset" in query_lower:
        response = (
            "To reset your password, go to the 'Login' page on our website, click 'Forgot Password,' and follow the steps to receive a reset link."
        )
        needs_escalation = False
    elif "locked" in query_lower or "suspended" in query_lower:
        response = (
            "If your account is locked or suspended, please check your email for details or provide more information for assistance."
        )
        needs_escalation = True
    elif "update" in query_lower or "change" in query_lower:
        response = (
            "To update your account details (e.g., phone number, email, address), log in and navigate to the 'Profile' section on our website or app."
        )
        needs_escalation = False
    elif "access certain features" in query_lower:
        response = (
            "If you can't access certain features, ensure your account is verified and meets the requirements. "
            "Check the 'Help' section or provide more details."
        )
        needs_escalation = state["sentiment"] == "Negative"
    elif "delete my account" in query_lower or "remove payment information" in query_lower:
        response = (
            "To delete your account or remove payment information, visit the 'Account Settings' section and follow the instructions."
        )
        needs_escalation = False
    else:
        response = (
            "For account-related issues, visit the 'Account' section on our website or provide more details for assistance."
        )
        needs_escalation = state["sentiment"] == "Negative"
    return {"response": response, "needs_escalation": needs_escalation}

def handle_promotions(state: State) -> State:
    query_lower = state["query"].lower()
    order_id = state.get("order_id", "None")
    if "apply a voucher" in query_lower or "promo code" in query_lower:
        response = (
            f"To apply a voucher or promo code to Order ID: {order_id}, enter the code at checkout in the 'Promotions' section on our website or app."
        )
        needs_escalation = False
    elif "voucher" in query_lower and ("not working" in query_lower or "invalid" in query_lower):
        response = (
            f"We're sorry your voucher for Order ID: {order_id} isn't working. Ensure the code is valid and meets the terms. "
            "Try code WELCOME10 for a 10% discount on your next order."
        )
        needs_escalation = state["sentiment"] == "Negative"
    elif "terms and conditions" in query_lower:
        response = (
            "Promotion terms are listed in the 'Offers' section on our website. Ensure your order meets the criteria (e.g., minimum spend, validity)."
        )
        needs_escalation = False
    elif "multiple vouchers" in query_lower:
        response = (
            "Currently, only one voucher or discount can be applied per order (Order ID: {order_id}). Check the 'Offers' section for details."
        )
        needs_escalation = False
    elif "eligible" in query_lower:
        response = (
            "To check promotion eligibility for Order ID: {order_id}, review the terms in the 'Offers' section or verify your account status."
        )
        needs_escalation = False
    elif "cashback" in query_lower or "discount not applied" in query_lower:
        response = (
            f"If your cashback or discount for Order ID: {order_id} was not applied, ensure the promotion was valid at checkout. "
            "Please provide more details for assistance."
        )
        needs_escalation = state["sentiment"] == "Negative"
    else:
        response = (
            f"For promotion inquiries for Order ID: {order_id}, visit the 'Offers' section on our website or provide more details."
        )
        needs_escalation = state["sentiment"] == "Negative"
    return {"response": response, "needs_escalation": needs_escalation}

def handle_general(state: State) -> State:
    query_lower = state["query"].lower()
    if "services" in query_lower:
        response = (
            "We offer a wide range of Aarong clothing, including traditional and modern apparel. "
            "Explore all products on our website or app."
        )
        needs_escalation = False
    elif "place an order" in query_lower or "transaction" in query_lower:
        response = (
            "To place an order, select your clothing items on our website or app, add to cart, and proceed to checkout."
        )
        needs_escalation = False
    elif "operating hours" in query_lower or "service availability" in query_lower:
        response = (
            "Our online store is available 24/7. Physical stores are open from 10 AM to 8 PM daily."
        )
        needs_escalation = False
    elif "contact customer support" in query_lower:
        response = (
            "You can reach us via this chatbot, email at support@aarong.com, or call our helpline at +880-123-456-7890."
        )
        needs_escalation = False
    elif "service available" in query_lower or "delivery areas" in query_lower:
        response = (
            "Check available delivery areas in the 'Shipping Info' section on our website or app."
        )
        needs_escalation = False
    elif "terms and conditions" in query_lower:
        response = (
            "Our terms and conditions are available in the 'Terms' section on our website. Please review them for details."
        )
        needs_escalation = False
    else:
        response = (
            "Thank you for reaching out! Please provide more details about your query, and we'll assist you promptly."
        )
        needs_escalation = state["sentiment"] == "Negative"
    return {"response": response, "needs_escalation": needs_escalation}

def handle_escalation(state: State) -> State:
    query_lower = state["query"].lower()
    order_id = state.get("order_id", "None")
    if "live agent" in query_lower or "escalate" in query_lower:
        dataset = load_dataset()
        agent = next((a for a in dataset["agents"] if "escalation" in a["specialty"].lower()), dataset["agents"][0])
        response = (
            f"Your query has been escalated to {agent['name']} (specialty: {agent['specialty']}). "
            f"Please contact them at {agent['contact_number']} with your order ID and details."
        )
        needs_escalation = False
    elif "how long" in query_lower and "respond" in query_lower:
        response = (
            f"A customer service representative will respond within 24-48 hours for Order ID: {order_id}. "
            "Please provide your order ID when contacted."
        )
        needs_escalation = False
    else:
        dataset = load_dataset()
        agent = next((a for a in dataset["agents"] if "escalation" in a["specialty"].lower()), dataset["agents"][0])
        response = (
            f"Your query has been escalated to {agent['name']} (specialty: {agent['specialty']}). "
            f"Please contact them at {agent['contact_number']} with your order ID and details."
        )
        needs_escalation = False
    return {"response": response, "needs_escalation": needs_escalation}

def escalate(state: State) -> State:
    dataset = load_dataset()
    category = state["category"].lower()
    agent = next((a for a in dataset["agents"] if category in a["specialty"].lower()), dataset["agents"][0])
    response = (
        f"Your query has been escalated to {agent['name']} (specialty: {agent['specialty']}). "
        f"Please contact them at {agent['contact_number']} with your order ID and details."
    )
    return {"response": response, "needs_escalation": False}

def route_query(state: State) -> str:
    if state.get("needs_escalation", False) or state["query"].lower().find("live agent") != -1:
        return "escalate"
    if state["category"] == "Default":
        return "update_conversation_history"
    category = state["category"].lower()
    if "complaints" in category:
        return "handle_complaints"
    elif "refunds" in category:
        return "handle_refunds"
    elif "delivery" in category:
        return "handle_delivery"
    elif "payments" in category:
        return "handle_payments"
    elif "account" in category:
        return "handle_account"
    elif "promotions" in category:
        return "handle_promotions"
    elif "escalation" in category:
        return "handle_escalation"
    else:
        return "handle_general"

workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("extract_order_id", extract_order_id)
workflow.add_node("update_conversation_history", update_conversation_history)
workflow.add_node("handle_complaints", handle_complaints)
workflow.add_node("handle_refunds", handle_refunds)
workflow.add_node("handle_delivery", handle_delivery)
workflow.add_node("handle_payments", handle_payments)
workflow.add_node("handle_account", handle_account)
workflow.add_node("handle_promotions", handle_promotions)
workflow.add_node("handle_general", handle_general)
workflow.add_node("handle_escalation", handle_escalation)
workflow.add_node("escalate", escalate)
workflow.set_entry_point("categorize")
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_edge("analyze_sentiment", "extract_order_id")
workflow.add_conditional_edges(
    "extract_order_id",
    route_query,
    {
        "handle_complaints": "handle_complaints",
        "handle_refunds": "handle_refunds",
        "handle_delivery": "handle_delivery",
        "handle_payments": "handle_payments",
        "handle_account": "handle_account",
        "handle_promotions": "handle_promotions",
        "handle_general": "handle_general",
        "handle_escalation": "handle_escalation",
        "escalate": "escalate",
        "update_conversation_history": "update_conversation_history"
    }
)
workflow.add_conditional_edges(
    "handle_complaints",
    lambda state: "escalate" if state["needs_escalation"] else "update_conversation_history",
    {"escalate": "escalate", "update_conversation_history": "update_conversation_history"}
)
workflow.add_conditional_edges(
    "handle_refunds",
    lambda state: "escalate" if state["needs_escalation"] else "update_conversation_history",
    {"escalate": "escalate", "update_conversation_history": "update_conversation_history"}
)
workflow.add_conditional_edges(
    "handle_delivery",
    lambda state: "escalate" if state["needs_escalation"] else "update_conversation_history",
    {"escalate": "escalate", "update_conversation_history": "update_conversation_history"}
)
workflow.add_conditional_edges(
    "handle_payments",
    lambda state: "escalate" if state["needs_escalation"] else "update_conversation_history",
    {"escalate": "escalate", "update_conversation_history": "update_conversation_history"}
)
workflow.add_conditional_edges(
    "handle_account",
    lambda state: "escalate" if state["needs_escalation"] else "update_conversation_history",
    {"escalate": "escalate", "update_conversation_history": "update_conversation_history"}
)
workflow.add_conditional_edges(
    "handle_promotions",
    lambda state: "escalate" if state["needs_escalation"] else "update_conversation_history",
    {"escalate": "escalate", "update_conversation_history": "update_conversation_history"}
)
workflow.add_conditional_edges(
    "handle_general",
    lambda state: "escalate" if state["needs_escalation"] else "update_conversation_history",
    {"escalate": "escalate", "update_conversation_history": "update_conversation_history"}
)
workflow.add_conditional_edges(
    "handle_escalation",
    lambda state: "escalate" if state["needs_escalation"] else "update_conversation_history",
    {"escalate": "escalate", "update_conversation_history": "update_conversation_history"}
)
workflow.add_edge("escalate", "update_conversation_history")
workflow.add_edge("update_conversation_history", END)
app = workflow.compile()

def run_customer_support(query: str, session_id: str, conversation_history: List[Dict[str, str]]) -> Dict[str, str]:
    result = app.invoke({
        "query": query,
        "order_id": "None",
        "needs_escalation": False,
        "session_id": session_id,
        "conversation_history": conversation_history
    })
    return {
        "category": result["category"],
        "sentiment": result.get("sentiment", "Neutral"),
        "response": result["response"],
        "order_id": result["order_id"],
        "session_id": result["session_id"],
        "conversation_history": result["conversation_history"]
    }

# Recommender page function
def recommender_page_function():
    st.markdown("""
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #1a202c;
                color: #e2e8f0;
            }
            .header {
                background-color: #2d3748;
                color: #e2e8f0;
                padding: 1.5rem;
                text-align: center;
                border-radius: 0.5rem;
                margin-bottom: 2rem;
            }
            .select-box {
                max-width: 600px;
                margin: 0 auto;
                padding: 1.5rem;
                background-color: #2d3748;
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .recommendation-card {
                background-color: #2d3748;
                border-radius: 0.5rem;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                transition: transform 0.2s;
                display: flex;
                align-items: center;
                gap: 1rem;
            }
            .recommendation-card:hover {
                transform: translateY(-5px);
            }
            .recommendation-image {
                width: 100px;
                height: 100px;
                object-fit: cover;
                border-radius: 0.25rem;
            }
            .error-message {
                background-color: #7f1d1d;
                color: #f3e8e8;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }
            .stButton>button {
                background-color: #4a5568;
                color: #e2e8f0;
                padding: 0.5rem 1.5rem;
                border-radius: 0.5rem;
                font-weight: bold;
                transition: background-color 0.2s;
            }
            .stButton>button:hover {
                background-color: #718096;
            }
            .stSelectbox select {
                border-radius: 0.5rem;
                padding: 0.5rem;
                background-color: #2d3748;
                color: #e2e8f0;
                border: 1px solid #4a5568;
            }
            .stSelectbox select option {
                background-color: #2d3748;
                color: #e2e8f0;
            }
            a {
                color: #60a5fa;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    """, unsafe_allow_html=True)

    MODEL_DIR = "Recommendation System Models"
    @st.cache_resource
    def load_data_and_models():
        try:
            df = pd.read_pickle(os.path.join(MODEL_DIR, "processed_data.pkl"))
            word2vec_model = Word2Vec.load(os.path.join(MODEL_DIR, "word2vec_model.model"))
            similarity_matrix = np.load(os.path.join(MODEL_DIR, "similarity_matrix.npy"))
            return df, word2vec_model, similarity_matrix
        except FileNotFoundError as e:
            st.error(f"Error: {e}. Please ensure 'processed_data.pkl', 'word2vec_model.model', and 'similarity_matrix.npy' are in the '{MODEL_DIR}' directory.")
            st.stop()
    new_df, word2vec_model, similarity = load_data_and_models()

    def recommend(cloth, df, similarity_matrix):
        try:
            cloth_index = df[df['Product Name'] == cloth].index[0]
            distances = similarity_matrix[cloth_index]
            cloth_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:10]
            recommended = []
            for i in cloth_list:
                if df.iloc[i[0]]['Product Name'] != cloth:
                    recommended.append(i)
                if len(recommended) >= 5:
                    break
            if not recommended:
                return None, "No recommendations found with different product names."
            results = []
            for i in recommended:
                product = df.iloc[i[0]]['Product Name']
                link = df.iloc[i[0]]['Link']
                image_url = "https://www.aarong.com/media/catalog/product/0/8/0870000089259_2.jpg?optimize=high&bg-color=255,255,255&fit=bounds&height=667&width=500&canvas=500:667"
                results.append((product, link, image_url))
            return results, None
        except IndexError:
            return None, f"Product '{cloth}' not found in the dataset."

    st.markdown("""
        <div class="header">
            <h1 class="text-3xl font-bold">Aarong Clothing Recommender</h1>
            <p class="text-lg">Discover similar clothing items from Aarong's collection</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("### Select a Product", unsafe_allow_html=True)
    product_names = new_df['Product Name'].tolist()
    selected_product = st.selectbox(
        "Select a product",
        product_names,
        placeholder="Choose a product...",
        label_visibility="collapsed"
    )
    if st.button("Get Recommendations"):
        recommendations, error = recommend(selected_product, new_df, similarity)
        if error:
            st.markdown(f'<div class="error-message">{error}</div>', unsafe_allow_html=True)
        else:
            st.markdown("### Recommended Products", unsafe_allow_html=True)
            for i, (product, link, image_url) in enumerate(recommendations, 1):
                st.markdown(f"""
                    <div class="recommendation-card">
                        <img src="{image_url}" class="recommendation-image">
                        <div>
                            <h3 class="text-lg font-semibold">{i}. {product}</h3>
                            <a href="{link}" target="_blank" class="text-blue-400 hover:underline">View Product</a>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# Forecasting page function
def forecast_page_function():
    st.markdown("""
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #1a202c;
                color: #e2e8f0;
            }
            .header {
                background-color: #2d3748;
                color: #e2e8f0;
                padding: 1.5rem;
                text-align: center;
                border-radius: 0.5rem;
                margin-bottom: 2rem;
            }
            .select-box {
                max-width: 600px;
                margin: 0 auto;
                padding: 1.5rem;
                background-color: #2d3748;
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .highlight {
                background-color: #4a5568;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)
    MODEL_DIR = "Recommendation System Models"
    csv_path = os.path.join(MODEL_DIR, "PyTorch_LSTM_GRU_Forecast.csv")
    try:
        df = pd.read_csv(csv_path)
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['yhat'] = pd.to_numeric(df['yhat'], errors='coerce').fillna(0).round(2)
        df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
        df['Product Name'] = df['Product Name'].str.strip()
        df = df[df['ds'].notnull() & (df['yhat'] != 0) & df['Product Name'].notnull() & (df['week'] != 0)]
    except FileNotFoundError:
        st.error(f"Error: 'PyTorch_LSTM_GRU_Forecast.csv' not found in '{MODEL_DIR}' directory.")
        return
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return
    st.markdown("""
        <div class="header">
            <h1 class="text-3xl font-bold">Aarong Weekly Demand Forecasting</h1>
            <p class="text-lg">Explore weekly demand trends for Aarong products</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("### Select Products", unsafe_allow_html=True)
    all_products = sorted(df['Product Name'].unique())
    initial_demands = df.groupby('Product Name').first()['yhat'].sort_values(ascending=False)
    default_products = initial_demands.head(5).index.tolist()
    selected_products = st.multiselect(
        "Select products",
        all_products,
        default=default_products,
        placeholder="Choose products...",
        label_visibility="collapsed"
    )
    max_drop = 0
    max_drop_product = ''
    for product in all_products:
        product_data = df[df['Product Name'] == product]['yhat']
        if not product_data.empty:
            drop = product_data.max() - product_data.min()
            if drop > max_drop:
                max_drop = drop
                max_drop_product = product
    insight = (f"Insight: '{max_drop_product}' shows the largest demand drop of {max_drop:.2f} units over the 14 weeks, "
               f"indicating a potential need for inventory adjustment.") if max_drop_product else "No significant demand drop detected."
    st.markdown("""
        <div class="highlight">
            <h2 class="text-xl font-bold mb-2">Interesting Insight</h2>
            <p>{}</p>
        </div>
    """.format(insight), unsafe_allow_html=True)
    if selected_products:
        plot_df = df[df['Product Name'].isin(selected_products)][['week', 'yhat', 'Product Name']].sort_values('week')
        fig = px.line(
            plot_df,
            x='week',
            y='yhat',
            color='Product Name',
            title='Weekly Demand Forecast',
            labels={'week': 'Week', 'yhat': 'Demand (Units)', 'Product Name': 'Product'},
            template='plotly_dark'
        )
        fig.update_layout(
            xaxis_range=[1, 14],
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='#2d3748',
            paper_bgcolor='#2d3748',
            font_color='#e2e8f0',
            showlegend=True,
            legend=dict(font=dict(size=12)),
            xaxis=dict(tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12))
        )
        fig.update_traces(line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one product to display the forecast.")

# Customer Support Chatbot page function
def chatbot_page_function():
    st.markdown("""
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #1a202c;
                color: #e2e8f0;
            }
            .header {
                background-color: #2d3748;
                color: #e2e8f0;
                padding: 1.5rem;
                text-align: center;
                border-radius: 0.5rem;
                margin-bottom: 2rem;
            }
            .chat-container {
                max-width: 600px;
                margin: 0 auto;
                background-color: #2d3748;
                border-radius: 0.5rem;
                padding: 1.5rem;
                height: 400px;
                overflow-y: auto;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .chat-message {
                margin-bottom: 1rem;
                padding: 1rem;
                border-radius: 0.5rem;
            }
            .user-message {
                background-color: #4a5568;
                text-align: right;
            }
            .bot-message {
                background-color: #718096;
                text-align: left;
            }
            .input-container {
                max-width: 600px;
                margin: 1rem auto;
                display: flex;
                gap: 1rem;
            }
            .stTextInput input {
                border-radius: 0.5rem;
                padding: 0.5rem;
                background-color: #2d3748;
                color: #e2e8f0;
                border: 1px solid #4a5568;
            }
            .stButton>button {
                background-color: #4a5568;
                color: #e2e8f0;
                padding: 0.5rem 1.5rem;
                border-radius: 0.5rem;
                font-weight: bold;
                transition: background-color 0.2s;
            }
            .stButton>button:hover {
                background-color: #718096;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="header">
            <h1 class="text-3xl font-bold">Aarong Customer Support Chatbot</h1>
            <p class="text-lg">Get instant help with your shopping queries</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'chat_session_id' not in st.session_state:
        st.session_state.chat_session_id = str(uuid4())
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Chat display
    chat_container = st.container()
    with chat_container:
        st.markdown(
    '''
    <div style="text-align: center;">
        <img src="https://img.freepik.com/free-vector/flat-design-illustration-customer-support_23-2148887720.jpg"
             style="width: 740px; height: 500px; object-fit: contain; margin-bottom: 1rem;">
    </div>
    ''',
    unsafe_allow_html=True
)
        for message in st.session_state.conversation_history:
            if message['query']:
                st.markdown(f'<div class="chat-message user-message">You: {message["query"]}</div>', unsafe_allow_html=True)
            if message['response']:
                st.markdown(f'<div class="chat-message bot-message">Bot: {message["response"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        query = st.text_input("Type your query here...", placeholder="Ask about orders, refunds, delivery, etc.", label_visibility="collapsed")
        submit_button = st.form_submit_button("Send")
        st.markdown('</div>', unsafe_allow_html=True)

        if submit_button and query.strip():
            if query.lower() in ["exit", "quit"]:
                st.session_state.conversation_history.append({"query": query, "response": "Thank you for using our support service. Goodbye!", "timestamp": str(datetime.now())})
                st.session_state.chat_session_id = str(uuid4())  # Reset session
                st.session_state.conversation_history = []
                st.rerun()
            else:
                result = run_customer_support(query, st.session_state.chat_session_id, st.session_state.conversation_history)
                st.session_state.conversation_history = result["conversation_history"]
                st.rerun()

# Sidebar with radio button navigation
st.sidebar.markdown("""
    <div class="p-4">
        <h2 class="text-xl font-bold mb-4 text-gray-200">Aarong Clothing App</h2>
    </div>
""", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Select Page",
    ["Clothing Recommender", "Demand Forecasting", "Customer Support Chatbot"],
    index=0,
    format_func=lambda x: x
)
st.sidebar.markdown("""
    <div class="p-4">
        <h2 class="text-xl font-bold mb-4 text-gray-200">Model Information</h2>
        <p class="text-gray-400">File sizes:</p>
""", unsafe_allow_html=True)
MODEL_DIR = "Recommendation System Models"
files = ["processed_data.pkl", "word2vec_model.model", "similarity_matrix.npy", "PyTorch_LSTM_GRU_Forecast.csv"]
for file in files:
    file_path = os.path.join(MODEL_DIR, file)
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        st.sidebar.markdown(f"- {file}: {size_mb:.2f} MB", unsafe_allow_html=True)
    except FileNotFoundError:
        st.sidebar.markdown(f"- {file}: Not found", unsafe_allow_html=True)
if os.path.exists("support_dataset.json"):
    try:
        size_bytes = os.path.getsize("support_dataset.json")
        size_mb = size_bytes / (1024 * 1024)
        st.sidebar.markdown(f"- support_dataset.json: {size_mb:.2f} MB", unsafe_allow_html=True)
    except:
        st.sidebar.markdown("- support_dataset.json: Not found", unsafe_allow_html=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Page rendering
if page == "Clothing Recommender":
    recommender_page_function()
elif page == "Demand Forecasting":
    forecast_page_function()
elif page == "Customer Support Chatbot":
    chatbot_page_function()