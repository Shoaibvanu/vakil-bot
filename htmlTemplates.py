css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color:#475063 
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 40%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}

.css-1aehpvj{
    display:none;
}



'''



bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/G7PB1tX/vakil-logo.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover" >
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <a><img src="https://img.freepik.com/free-vector/businessman-character-avatar-isolated_24877-60111.jpg?w=740&t=st=1705485942~exp=1705486542~hmac=4a1088e6d07c89f269f6cc12484f14fd37c7f8d0dd47a80674f1508b87019cfc" 
        alt="vakil-logo" border="0" style="height=20px;width=20px;color: #DED8C4"></a><br /><a target='_blank'></a><br />
    </div>    
    <div class="message">{{MSG}}</div>
</div>

'''
