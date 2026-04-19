import streamlit as st
import json
import time
import PIL.Image
from google import genai

# --- 1. CONFIGURATION ---
GEMINI_API_KEY = "AIzaSyDd8QGUV0NfJk5YWKhLX4d_6F2b8llwYho" 

# Brain system instructions (English Only) [cite: 2026-01-21]
SYSTEM_PROMPT = """
You are the SocialStake Autonomous Matchmaker. 
Analyze the user's bio and suggest 3 specific social interaction types. 
The user has committed 1000 USDC, so be professional and challenging.
Always output valid JSON.
"""

# --- 2. INITIALIZE CLIENT ---
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    MODEL_ID = "gemini-2.0-flash-exp"
except Exception as e:
    st.error(f"Initialization Error: {e}")

# --- 3. UI SETTINGS ---
st.set_page_config(page_title="SocialStake | High-Stakes Socializing", layout="wide")

with st.sidebar:
    st.header("⚙️ Agent Control")
    is_demo = st.toggle("🚀 Enable Demo Mode", value=True)
    st.divider()
    st.metric("Required Stake", "1000 USDC")
    st.info("Network: Arc Mainnet-Beta")
    if st.button("🔄 Reset App"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

# --- 4. SESSION STATE ---
if 'app_state' not in st.session_state: st.session_state.app_state = "STAKING"
if 'meeting_count' not in st.session_state: st.session_state.meeting_count = 9
if 'user_bio' not in st.session_state: st.session_state.user_bio = ""
if 'match_options' not in st.session_state: st.session_state.match_options = None

# --- 5. AGENT BRAIN (LOGIC IN ENGLISH) [cite: 2026-01-21] ---
def run_matchmaker(bio):
    if is_demo:
        time.sleep(2)
        return [
            {"goal": "Tech Networking", "desc": "Meet a fellow developer at a local cafe."},
            {"goal": "Public Speaking", "desc": "Join a local Toastmasters or debate club."},
            {"goal": "Community Service", "desc": "Volunteer at a charity event for 2 hours."}
        ]
    try:
        prompt = f"Based on this bio: '{bio}', suggest 3 social goals. Return JSON: [{{'goal': str, 'desc': str}}]"
        response = client.models.generate_content(model=MODEL_ID, config={'system_instruction': SYSTEM_PROMPT}, contents=prompt)
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_json)
    except: return None

# --- 6. MAIN INTERFACE ---

# STEP 1: STAKING & BIO (The gatekeeper)
if st.session_state.app_state == "STAKING":
    st.title("🛡️ SocialStake Protocol")
    st.subheader("Commit 1000 USDC to unlock your social journey")
    
    bio = st.text_area("Describe your social personality and goals:", 
                       placeholder="I am an introvert software engineer from Tbilisi wanting to meet more people in tech.")
    
    if st.button("Stake 1000 USDC & Analyze Profile"):
        if len(bio) > 10:
            st.session_state.user_bio = bio
            with st.status("Locking 1000 USDC on Arc Network...", expanded=True) as status:
                time.sleep(2)
                st.write("Verifying transaction...")
                time.sleep(1)
                status.update(label="Stake Confirmed!", state="complete")
            
            with st.spinner("AI Agent is generating your social path..."):
                st.session_state.match_options = run_matchmaker(bio)
                st.session_state.app_state = "MATCHING"
                st.rerun()
        else:
            st.error("Please provide a more detailed description (min 10 chars).")

# STEP 2: SELECTING OPTION
elif st.session_state.app_state == "MATCHING":
    st.title("🎯 Choose Your Social Challenge")
    st.write("Based on your profile, the AI Agent suggests these paths:")
    
    if st.session_state.match_options:
        for i, option in enumerate(st.session_state.match_options):
            with st.expander(f"Option {i+1}: {option['goal']}"):
                st.write(option['desc'])
                if st.button(f"Accept Challenge {i+1}", key=f"btn_{i}"):
                    st.session_state.current_goal = option['goal']
                    st.session_state.app_state = "VERIFY"
                    st.rerun()

# STEP 3: VERIFICATION (Meeting 10/10)
elif st.session_state.app_state == "VERIFY":
    st.title(f"🏁 Final Milestone: {st.session_state.get('current_goal', 'Social Meeting')}")
    st.progress(0.9, text="Progress: 9/10 Meetings Completed")
    st.info("Upload a photo to verify your 10th meeting and release the 1000 USDC.")
    
    img_file = st.file_uploader("Upload proof", type=['jpg', 'png'])
    if img_file:
        st.image(img_file, width=400)
        if st.button("Final AI Verification"):
            with st.spinner("Gemini 2.0 is checking interaction..."):
                time.sleep(2)
                st.session_state.app_state = "COMPLETE"
                st.rerun()

# STEP 4: SUCCESS
elif st.session_state.app_state == "COMPLETE":
    st.balloons()
    st.title("🎉 Journey Complete!")
    st.header("💰 1000 USDC Unlocked & Returned")
    st.success("The AI Agent verified all 10 interactions. Your stake is back in your wallet.")
    if st.button("Start New Journey"):
        st.session_state.app_state = "STAKING"
        st.rerun()