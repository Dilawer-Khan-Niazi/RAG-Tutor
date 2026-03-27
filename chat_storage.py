import json
import os
from datetime import datetime
from typing import List, Dict
import uuid

class ChatStorage:
    """Enhanced chat storage with book tracking."""
    
    def __init__(self, storage_dir: str = "chat_history"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.sessions_file = os.path.join(storage_dir, "sessions.json")
        
        if os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                self.sessions = json.load(f)
        else:
            self.sessions = {}
            self._save_sessions()
    
    def _save_sessions(self):
        with open(self.sessions_file, 'w', encoding='utf-8') as f:
            json.dump(self.sessions, f, indent=2, ensure_ascii=False)
    
    def create_session(self, book_id: str = None, book_title: str = None) -> str:
        session_id = str(uuid.uuid4())[:8]
        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "book_id": book_id,
            "book_title": book_title,
            "messages": [],
            "title": "New Chat"
        }
        self._save_sessions()
        return session_id
    
    def add_message(
        self,
        session_id: str,
        question: str,
        answer: str,
        difficulty: str,
        agent_log: Dict = None,
        book_id: str = None
    ):
        if session_id not in self.sessions:
            session_id = self.create_session(book_id=book_id)
        
        message = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "difficulty": difficulty,
            "agent_log": agent_log,
            "book_id": book_id
        }
        
        self.sessions[session_id]["messages"].append(message)
        
        if len(self.sessions[session_id]["messages"]) == 1:
            self.sessions[session_id]["title"] = question[:50] + ("..." if len(question) > 50 else "")
        
        self._save_sessions()
        return session_id
    
    def get_session(self, session_id: str) -> Dict:
        return self.sessions.get(session_id, None)
    
    def get_all_sessions(self) -> List[Dict]:
        sessions = list(self.sessions.values())
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions
    
    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
    
    def get_chat_history_html(self, session_id: str) -> str:
        """Format chat history as clean HTML."""
        session = self.get_session(session_id)
        if not session:
            return "<p>No chat history found.</p>"
        
        html = f"""
        <div style='max-width: 850px; margin: 0 auto; font-family: Inter, sans-serif;'>
            <h2 style='color: #111827; margin-bottom: 5px;'>{session['title']}</h2>
            <p style='color: #4b5563; font-size: 0.88em; margin-top: 2px; margin-bottom: 12px;'>
                Book: {session.get('book_title', 'Unknown')} | Created: {session['created_at'][:19]}
            </p>
            <hr style='border: 1px solid #e5e7eb; margin: 12px 0;'>
        """
        
        for i, msg in enumerate(session["messages"], 1):
            # User message
            html += f"""
            <div style='background: #ffffff; padding: 12px; margin: 10px 0;'>
                <div style='font-weight: 600; color: #111827; margin-bottom: 5px;'>
                    You
                </div>
                <div style='color: #111827;'>
                    {msg['question']}
                </div>
                <div style='font-size: 0.81em; color: #6b7280; margin-top: 5px;'>
                    {msg['timestamp'][:19]} | Difficulty: {msg['difficulty']}
                </div>
            </div>
            """
            
            # AI response
            html += f"""
            <div style='background: #f9fafb; padding: 12px; margin: 10px 0 24px 0;'>
                <div style='font-weight: 600; color: #111827; margin-bottom: 5px;'>
                    AI Tutor
                </div>
                <div style='color: #111827; line-height: 1.5;'>
                    {msg['answer'].replace(chr(10), '<br>')}
                </div>
            """
            
            # Agent log if available
            if msg.get('agent_log'):
                html += f"""
                <details style='margin-top: 10px; font-size: 0.9em; color: #7f8c8d;'>
                    <summary style='cursor: pointer;'>🔍 Show retrieval details</summary>
                    <div style='margin-top: 8px; padding: 10px; background: white; border-radius: 5px;'>
                        Iterations: {len(msg['agent_log']['iterations'])} | 
                        Chunks: {msg['agent_log']['total_chunks_retrieved']} | 
                        Confidence: {msg['agent_log']['final_confidence']:.0%}
                    </div>
                </details>
                """
            
            html += "</div>"
        
        html += "</div>"
        return html