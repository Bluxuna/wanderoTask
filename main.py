import time
import json
import os
from typing import Dict, List, Any, TypedDict, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
import aiosmtplib
from imapclient import IMAPClient
from mailparser import parse_from_bytes
from email.mime.text import MIMEText
import ssl
import asyncio
from langgraph.graph import StateGraph, END
import streamlit as st
from datetime import datetime
import random
import logging

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Wandero Email Client Simulator",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Logging Setup ---
logging.basicConfig(
    filename='wandero_simulation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Email Configuration from .env file ---
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER", "your_email@example.com")
EMAIL_PASS = os.getenv("EMAIL_PASS", "your_email_password")
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", 993))


# --- Groq Client Initialization ---
@st.cache_resource
def initialize_groq_client():
    try:
        client = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192"
        )
        return client, True
    except Exception as e:
        return None, str(e)


groq_client, groq_status = initialize_groq_client()


# --- State Structure Definition ---
class EmailWorkflowState(TypedDict):
    persona: Dict[str, Any]
    chat_history: List[BaseMessage]
    recipient_email: str
    last_sent_email: Optional[Dict]
    last_received_email: Optional[Dict]
    current_step: str
    is_finished: bool
    simulation_id: str
    response_times: List[float]
    success_status: str


# --- Email Functions ---
async def send_email(to_email: str, subject: str, body: str) -> Optional[Dict]:
    """Sends email and returns details on success."""
    message = MIMEText(body)
    message["From"] = EMAIL_USER
    message["To"] = to_email
    message["Subject"] = subject

    try:
        await aiosmtplib.send(
            message,
            hostname=EMAIL_HOST,
            port=EMAIL_PORT,
            username=EMAIL_USER,
            password=EMAIL_PASS,
            use_tls=True
        )
        return {
            "to": to_email,
            "subject": subject,
            "body": body,
            "timestamp": time.time(),
        }
    except Exception as e:
        st.error(f"Email sending error: {e}")
        logger.error(f"Email sending error: {e}")
        return None


def fetch_latest_reply(from_address: str) -> Optional[Dict]:
    """Checks mailbox for reply from specific address."""
    try:
        with IMAPClient(IMAP_HOST, port=IMAP_PORT, ssl=True) as client:
            client.login(EMAIL_USER, EMAIL_PASS)
            client.select_folder("INBOX")

            messages = client.search(["UNSEEN", "FROM", from_address])
            if not messages:
                return None

            latest_uid = messages[-1]
            for uid, data in client.fetch([latest_uid], ["RFC822"]).items():
                raw_email_bytes = data[b"RFC822"]
                mail = parse_from_bytes(raw_email_bytes)

                email_data = {
                    "from": mail.from_,
                    "subject": mail.subject,
                    "body": mail.text_plain[0] if mail.text_plain else "No body content",
                    "date": mail.date
                }
                client.add_flags(uid, ['\\Seen'])
                return email_data
    except Exception as e:
        st.error(f"Email checking error: {e}")
        logger.error(f"Email checking error: {e}")
        return None


# --- Workflow Nodes ---
async def generate_client_persona(state: EmailWorkflowState) -> EmailWorkflowState:
    """Generates the client persona and initial request with randomization."""
    company_info = st.session_state.get('company_info', "Travel company 'Wandero', which operates in Georgia.")
    company_country = "Georgia" if "Georgia" in company_info else "Unknown"
    travel_styles = ["Budget", "Luxury", "Adventure", "Family", "Culinary"]
    interests_pool = [
        "Hiking", "Wine Tasting", "Historical Sites", "Beach Relaxation",
        "Cultural Tours", "Food Tours", "Adventure Sports", "City Exploration"
    ]
    behaviors = ["decisive", "indecisive", "confused"]

    # Randomize persona details
    num_people = random.randint(1, 8)
    travel_dates = [
        f"2025-{random.randint(8, 12):02d}-{random.randint(1, 28):02d}",
        f"2025-{random.randint(8, 12):02d}-{random.randint(1, 28):02d}"
    ]
    edge_case = random.random() < 0.2  # 20% chance to request unsupported country
    destination = "Japan" if edge_case else company_country

    prompt = f"""
    Create a persona for a client who wants to travel.
    Information about the target company: {company_info}

    Generate the following information in JSON format:
    ```json
    {{
        "sender_name": "A realistic client name.",
        "travel_style": "{random.choice(travel_styles)}",
        "interests": ["{random.choice(interests_pool)}", "{random.choice(interests_pool)}"],
        "behavior": "{random.choice(behaviors)}",
        "num_people": {num_people},
        "travel_dates": ["{travel_dates[0]}", "{travel_dates[1]}"],
        "destination": "{destination}",
        "initial_request": "A short, natural initial inquiry for a tour in {destination} for {num_people} people on {travel_dates[0]} to {travel_dates[1]}.",
        "initial_subject": "Inquiry about {destination} Tour"
    }}
    ```
    Ensure the output is ONLY the JSON block, nothing else.
    """

    try:
        response = await groq_client.ainvoke([HumanMessage(content=prompt)])
        response_content = response.content.strip()

        if response_content.startswith("```json") and response_content.endswith("```"):
            json_string = response_content[7:-3].strip()
        else:
            json_string = response_content

        persona_data = json.loads(json_string)
        state["persona"] = persona_data
        state["chat_history"] = [HumanMessage(content=persona_data["initial_request"])]
        state["current_step"] = "persona_generated"
        state["response_times"] = []
        state["success_status"] = "in_progress"

        logger.info(f"Simulation {state['simulation_id']}: Persona generated: {persona_data['sender_name']}")
        with st.session_state['persona_placeholder']:
            st.success(f"✅ Generated Persona: {persona_data['sender_name']} ({persona_data['travel_style']})")
            st.json(persona_data)

    except json.JSONDecodeError as e:
        st.error(f"JSON decoding error: {e}")
        logger.error(f"Simulation {state['simulation_id']}: JSON decoding error: {e}")
        state["current_step"] = "error"
        state["is_finished"] = True
        state["success_status"] = "failed"
    except Exception as e:
        st.error(f"Error generating persona: {e}")
        logger.error(f"Simulation {state['simulation_id']}: Error generating persona: {e}")
        state["current_step"] = "error"
        state["is_finished"] = True
        state["success_status"] = "failed"

    return state


async def send_initial_request_node(state: EmailWorkflowState) -> EmailWorkflowState:
    """Sends initial request."""
    try:
        persona = state["persona"]
        sent_email = await send_email(
            to_email=state["recipient_email"],
            subject=persona["initial_subject"],
            body=persona["initial_request"]
        )

        if sent_email:
            state["last_sent_email"] = sent_email
            state["current_step"] = "initial_email_sent"
            logger.info(f"Simulation {state['simulation_id']}: Initial email sent to {state['recipient_email']}")
            with st.session_state['email_placeholder']:
                st.success(f"📧 Initial email sent to {state['recipient_email']}")
        else:
            state["current_step"] = "error"
            state["is_finished"] = True
            state["success_status"] = "failed"
            st.error("❌ Failed to send initial email")
            logger.error(f"Simulation {state['simulation_id']}: Failed to send initial email")
    except Exception as e:
        st.error(f"Error sending initial email: {e}")
        logger.error(f"Simulation {state['simulation_id']}: Error sending initial email: {e}")
        state["current_step"] = "error"
        state["is_finished"] = True
        state["success_status"] = "failed"

    return state


async def wait_and_analyze_reply_node(state: EmailWorkflowState) -> EmailWorkflowState:
    """Waits for reply, analyzes it, and generates response with behavior variability."""
    try:
        wait_time = st.session_state.get('wait_time', 60)
        start_time = time.time()

        with st.session_state['progress_placeholder']:
            for i in range(wait_time):
                st.info(f"⏳ Waiting for reply... ({i + 1}/{wait_time}s)")
                await asyncio.sleep(1)

        received_email = await asyncio.to_thread(fetch_latest_reply, state["recipient_email"])

        if not received_email:
            st.warning("⏰ Timeout reached, no reply received. Ending simulation.")
            logger.warning(f"Simulation {state['simulation_id']}: Timeout reached, no reply received")
            state["is_finished"] = True
            state["current_step"] = "timeout"
            state["success_status"] = "timeout"
            return state

        response_time = time.time() - start_time
        state["response_times"].append(response_time)
        state["last_received_email"] = received_email
        state["chat_history"].append(AIMessage(content=received_email["body"]))

        logger.info(f"Simulation {state['simulation_id']}: Received reply: {received_email['subject']}")
        with st.expander(f"📨 Received: {received_email['subject']}", expanded=True):
            st.write(received_email["body"])

        persona = state["persona"]
        chat_history_str = "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in state["chat_history"]])

        # Handle behavior variability
        behavior = persona.get("behavior", "decisive")
        follow_up_chance = random.random() < 0.3 if behavior == "indecisive" else 0.1
        if follow_up_chance and state["current_step"] != "initial_email_sent":
            prompt = f"""
            You are a {behavior} client named {persona['sender_name']}. You forgot to mention something in your last email.
            Add a follow-up detail related to your tour request (e.g., dietary restrictions, extra activity, or change in dates).
            Return ONLY the body of the follow-up email.
            """
            response = await groq_client.ainvoke([HumanMessage(content=prompt)])
            follow_up_text = response.content.strip()
            subject = f"Re: {received_email['subject']} - Additional Details"
            sent_email = await send_email(
                to_email=state["recipient_email"],
                subject=subject,
                body=follow_up_text
            )
            if sent_email:
                state["last_sent_email"] = sent_email
                state["chat_history"].append(HumanMessage(content=follow_up_text))
                state["current_step"] = "follow_up_sent"
                logger.info(f"Simulation {state['simulation_id']}: Sent follow-up email")
                with st.expander(f"📤 Sent: {subject}", expanded=True):
                    st.write(follow_up_text)
            return state

        # Detect invoice or final plan
        email_body = received_email["body"].lower()
        is_invoice = any(keyword in email_body for keyword in ["invoice", "payment", "cost breakdown"])
        is_final_plan = any(keyword in email_body for keyword in ["final plan", "confirmed itinerary", "final details"])

        if is_final_plan:
            prompt = f"""
            You are a {behavior} client named {persona['sender_name']}. The company sent a final tour plan: "{email_body}".
            Confirm the plan and thank the company for their service.
            Return ONLY the body of the confirmation email.
            """
            response = await groq_client.ainvoke([HumanMessage(content=prompt)])
            reply_text = response.content.strip()
            subject = f"Re: {received_email['subject']} - Confirmation"
            state["success_status"] = "completed"
        elif is_invoice:
            prompt = f"""
            You are a {behavior} client named {persona['sender_name']}. The company sent an invoice: "{email_body}".
            Confirm receipt of the invoice and simulate payment completion.
            Request the final tour plan.
            Return ONLY the body of the email response.
            """
            response = await groq_client.ainvoke([HumanMessage(content=prompt)])
            reply_text = response.content.strip()
            subject = f"Re: {received_email['subject']} - Payment Confirmation"
            state["current_step"] = "invoice_confirmed"
        else:
            prompt = f"""
            You are a {behavior} client named {persona['sender_name']}.
            Your Persona:
            - Travel Style: {persona['travel_style']}
            - Interests: {persona['interests']}
            - Destination: {persona['destination']}
            - Number of People: {persona['num_people']}
            - Travel Dates: {persona['travel_dates']}

            Conversation History:
            {chat_history_str}

            The company's latest reply: "{received_email['body']}"

            Your Task:
            Analyze the company's reply and write a natural email response.
            1. If the company asks questions, answer them based on your persona.
            2. If the company proposes a tour plan, evaluate it against your interests. 
               - If {behavior == 'indecisive'}, request at least one modification (e.g., change an activity or date).
               - If {behavior == 'confused'}, ask vague or clarifying questions.
               - If {behavior == 'decisive' and random.random() < 0.7}, accept the plan unless it conflicts with your interests.
            3. If satisfied and the plan is complete, confirm it.
            4. If the dialogue is complete, return ONLY: "END_CONVERSATION".

            Return ONLY the body of the email reply or "END_CONVERSATION".
            """

            response = await groq_client.ainvoke(state["chat_history"] + [HumanMessage(content=prompt)])
            reply_text = response.content.strip()

            if reply_text == "END_CONVERSATION":
                st.success("🏁 Agent decided to end the conversation.")
                logger.info(f"Simulation {state['simulation_id']}: Conversation ended by agent")
                state["is_finished"] = True
                state["current_step"] = "conversation_ended_by_agent"
                state["success_status"] = "completed"
                return state

            subject = f"Re: {received_email['subject']}"

        sent_email = await send_email(
            to_email=state["recipient_email"],
            subject=subject,
            body=reply_text
        )
        if sent_email:
            state["last_sent_email"] = sent_email
            state["chat_history"].append(HumanMessage(content=reply_text))
            state["current_step"] = "reply_sent"
            logger.info(f"Simulation {state['simulation_id']}: Reply sent: {subject}")
            with st.expander(f"📤 Sent: {subject}", expanded=True):
                st.write(reply_text)
        else:
            state["is_finished"] = True
            state["current_step"] = "error"
            state["success_status"] = "failed"
            st.error("❌ Failed to send reply")
            logger.error(f"Simulation {state['simulation_id']}: Failed to send reply")

    except Exception as e:
        st.error(f"Error in wait_and_analyze_reply: {e}")
        logger.error(f"Simulation {state['simulation_id']}: Error in wait_and_analyze_reply: {e}")
        state["is_finished"] = True
        state["current_step"] = "error"
        state["success_status"] = "failed"

    return state


async def should_continue_conversation(state: EmailWorkflowState) -> str:
    """Decides whether to continue conversation."""
    if state.get("is_finished", False):
        return END
    else:
        return "wait_and_analyze_reply_node"


# --- Build Workflow Graph ---
def build_workflow():
    workflow = StateGraph(EmailWorkflowState)

    workflow.add_node("generate_client_persona_node", generate_client_persona)
    workflow.add_node("send_initial_request_node", send_initial_request_node)
    workflow.add_node("wait_and_analyze_reply_node", wait_and_analyze_reply_node)

    workflow.set_entry_point("generate_client_persona_node")
    workflow.add_edge("generate_client_persona_node", "send_initial_request_node")
    workflow.add_edge("send_initial_request_node", "wait_and_analyze_reply_node")
    workflow.add_conditional_edges(
        "wait_and_analyze_reply_node",
        should_continue_conversation,
    )

    return workflow.compile()


# --- Parallel Simulation Function ---
async def run_parallel_simulations(wandero_email: str, num_simulations: int):
    """Runs multiple client simulations concurrently."""
    import uuid
    tasks = []
    for i in range(num_simulations):
        simulation_id = str(uuid.uuid4())
        initial_state: EmailWorkflowState = {
            "persona": {},
            "chat_history": [],
            "recipient_email": wandero_email,
            "last_sent_email": None,
            "last_received_email": None,
            "current_step": "initial",
            "is_finished": False,
            "simulation_id": simulation_id,
            "response_times": [],
            "success_status": "in_progress"
        }
        app = build_workflow()
        tasks.append(app.ainvoke(initial_state, {"recursion_limit": 15}))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Save results
    output_file = "company_data.json"

    def custom_serializer(obj):
        if isinstance(obj, BaseMessage):
            return {"type": type(obj).__name__, "content": obj.content}
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    try:
        existing_data = []
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    existing_data = []

        for result in results:
            if not isinstance(result, Exception):
                existing_data.append(json.loads(json.dumps(result, default=custom_serializer)))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, default=custom_serializer)

        st.success(f"✅ {len(results)} simulation(s) completed and saved to {output_file}")
        logger.info(f"Completed {len(results)} parallel simulations")

    except Exception as e:
        st.error(f"Error saving results: {e}")
        logger.error(f"Error saving parallel simulation results: {e}")

    return results


# --- Streamlit UI ---
def main():
    st.title("📧 Wandero Email Client Simulator")
    st.markdown("Simulate client interactions with your travel agency through automated email exchanges.")

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("📊 Status")
        if groq_status is True:
            st.success("✅ Groq API Connected")
        else:
            st.error(f"❌ Groq API Error: {groq_status}")

        # if EMAIL_USER != "your_email@example.com" and EMAIL_PASS != "your_email_password":
        #     st.success("✅ Email Config Loaded")
        # else:
        #     st.error("❌ Email Config Missing")

        st.divider()

        st.subheader("🎛️ Settings")
        wandero_email = st.text_input(
            "Wandero Email Address:",
            value=os.getenv("WANDERO_EMAIL", ""),
            help="Email address where simulation requests will be sent"
        )

        company_info = st.text_area(
            "Company Information:",
            value="Travel company 'Wandero', which operates in Georgia.",
            help="Information about your company for persona generation"
        )

        wait_time = st.slider(
            "Wait Time Between Emails (seconds):",
            min_value=30,
            max_value=300,
            value=60,
            help="How long to wait for replies"
        )

        num_simulations = st.slider(
            "Number of Parallel Simulations:",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of concurrent client simulations"
        )

        st.session_state['company_info'] = company_info
        st.session_state['wait_time'] = wait_time
        st.session_state['num_simulations'] = num_simulations

    # Main Content Area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🚀 Run Simulation")

        if not wandero_email:
            st.warning("⚠️ Please enter Wandero email address in the sidebar")
            return

        if groq_status is not True:
            st.error("❌ Cannot run simulation: Groq API not configured properly")
            return

        if st.button("▶️ Start Simulation", type="primary", use_container_width=True):
            if 'simulation_running' not in st.session_state or not st.session_state.get('simulation_running', False):
                st.session_state['simulation_running'] = True

                # Create placeholders for real-time updates
                st.session_state['persona_placeholder'] = st.empty()
                st.session_state['email_placeholder'] = st.empty()
                st.session_state['progress_placeholder'] = st.empty()

                simulation_container = st.container()

                with simulation_container:
                    with st.spinner(f"Running {num_simulations} simulation(s)..."):
                        try:
                            results = asyncio.run(run_parallel_simulations(wandero_email, num_simulations))

                            st.success(f"🎉 {num_simulations} simulation(s) completed successfully!")

                            # Display results
                            st.subheader("📋 Final Results")
                            for i, result in enumerate(results):
                                if isinstance(result, Exception):
                                    st.error(f"❌ Simulation {i + 1} failed: {str(result)}")
                                    continue
                                st.write(f"**Simulation {i + 1} (ID: {result['simulation_id']})**")
                                if result.get('persona'):
                                    st.json(result['persona'])
                                st.write(f"**Status**: {result['success_status']}")
                                if result.get('response_times'):
                                    avg_response = sum(result['response_times']) / len(result['response_times'])
                                    st.write(f"**Average Response Time**: {avg_response:.2f} seconds")

                                st.subheader(f"💬 Conversation History (Simulation {i + 1})")
                                if result.get('chat_history'):
                                    for j, message in enumerate(result.get('chat_history', [])):
                                        msg_type = "🤖 AI" if isinstance(message, AIMessage) else "👤 Human"
                                        with st.expander(f"{msg_type} Message {j + 1}"):
                                            st.write(message.content)
                                else:
                                    st.info("No conversation history available")

                        except Exception as e:
                            st.error(f"❌ Simulation failed: {str(e)}")
                            logger.error(f"Simulation failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                        finally:
                            st.session_state['simulation_running'] = False
            else:
                st.info("⏳ Simulation is already running. Please wait for it to complete.")

    with col2:
        st.header("📊 History and Analytics")

        if os.path.exists("company_data.json"):
            try:
                with open("company_data.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if data:
                    st.metric("Total Simulations", len(data))

                    # Calculate analytics
                    success_count = sum(1 for sim in data if sim.get('success_status') == 'completed')
                    timeout_count = sum(1 for sim in data if sim.get('success_status') == 'timeout')
                    failed_count = sum(1 for sim in data if sim.get('success_status') == 'failed')
                    avg_response_times = [sum(sim.get('response_times', [])) / len(sim['response_times'])
                                          for sim in data if sim.get('response_times')]
                    avg_response = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0

                    st.metric("Successful Simulations", success_count)
                    st.metric("Timed Out Simulations", timeout_count)
                    st.metric("Failed Simulations", failed_count)
                    st.metric("Average Response Time (s)", f"{avg_response:.2f}")

                    # Show recent simulations
                    for i, sim in enumerate(reversed(data[-5:])):  # Last 5 simulations
                        with st.expander(f"Simulation {len(data) - i} (ID: {sim['simulation_id']})"):
                            if 'persona' in sim and sim['persona']:
                                st.write(f"**Client**: {sim['persona'].get('sender_name', 'N/A')}")
                                st.write(f"**Style**: {sim['persona'].get('travel_style', 'N/A')}")
                                st.write(f"**Status**: {sim.get('success_status', 'N/A')}")
                                if sim.get('response_times'):
                                    avg_time = sum(sim['response_times']) / len(sim['response_times'])
                                    st.write(f"**Avg Response Time**: {avg_time:.2f} seconds")

            except Exception as e:
                st.error(f"Error loading history: {e}")
                logger.error(f"Error loading history: {e}")
        else:
            st.info("No simulation history yet")

        if st.button("🗑️ Clear History", help="Delete all simulation data"):
            if os.path.exists("company_data.json"):
                os.remove("company_data.json")
                st.success("History cleared!")
                logger.info("Simulation history cleared")
                st.rerun()

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌟 Wandero Email Client Simulator v1.2</p>
        <p>Built with Streamlit, LangChain, and LangGraph</p>
        <p> Made by Giorgi Makharoblidze</p>
        <p>#blux</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()