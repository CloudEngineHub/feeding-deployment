<template>
  <div class="page">
    <div class="tb">
      <div class="av"><img src="../assets/user_avatar.svg" alt="User"></div>
      <div>
        <div class="tb-n">{{ username }}</div>
        <div class="tb-s">Adjust how the robot is feeding you</div>
      </div>
    </div>

    <div class="bd talk-bd">
      <p class="talk-lbl">What would you like to edit?</p>
      <div class="talk-box">
        <textarea
          v-model="transcript"
          placeholder="Typing..."
          ref="textarea"
          class="talk-input"
        ></textarea>
        <button @click="toggleSpeechRecognition"
                class="icon-btn"
                :class="{ 'amber-ic': isRecognizing }"
                :title="isRecognizing ? 'stop voice' : 'voice'">
          <img alt="voice" src="../assets/voice.png">
        </button>
        <button @click="cleartheinput" class="icon-btn" title="clear">
          <img alt="clear" src="../assets/clear.png">
        </button>
        <button @click="sendToRosFromTextBox" class="icon-btn amber-ic" title="send">
          <img alt="send" src="../assets/send.png">
        </button>
      </div>
      <p class="voice-status" :class="{ empty: !voiceStatus }" aria-live="polite">
        <img alt="" src="../assets/voice.png">
        <span v-if="voiceStatus">{{ voiceStatus }}</span>
        <span v-else>&nbsp;</span>
      </p>

      <p class="talk-lbl">Robot's response</p>
      <div class="response-box">{{ customOrder || 'Waiting for the text response...' }}</div>

      <button class="btn sm ghost w100" style="margin-top:auto" @click="$router.push('/task_selection')">
        ← Task Selection
      </button>
    </div>
  </div>
</template>

<script>
import ROSLIB from 'roslib'
import { ROS_URL, USER} from '@/config/parameterConfig';
import { createDictation } from '@/utils/dictation';

export default {
  data () {
    return {
      ros: null,
      username: USER,
      listener: null,
      publisher: null,
      transcript: '',
      isRecognizing: false,
      customOrder: '',
      voiceStatus: '',
    }
  },
  mounted () {
    this.ros = new ROSLIB.Ros({ url: ROS_URL })
    this.initPublisher()
    this.initRosConnection()
    // Non-reactive on purpose: it holds a live SpeechRecognition object.
    this._dictation = createDictation({
      onText: (text) => { this.transcript += text },
      onStatus: (status) => { this.voiceStatus = status },
      onActive: (active) => { this.isRecognizing = active }
    })
  },
  beforeUnmount () {
    if (this._dictation) this._dictation.destroy()
  },
  beforeRouteLeave (to, from, next) {
    if (this._dictation) this._dictation.destroy()

    if (this.listener) {
      this.listener.unsubscribe();
      this.listener = null;
    }

    if (this.publisher) {
      this.publisher.unadvertise();
      this.publisher = null;
    }

    next();
  },
  methods: {
    sendToRosFromTextBox() {
      if (this.publisher && this.transcript.trim() !== '') {
        const message = new ROSLIB.Message({
          data: JSON.stringify({
            state: 'adaptability_request',
            status: this.transcript
          })
        });
        this.publisher.publish(message);
        this.transcript = '';
      } else {
      }
    },
    initRosConnection() {

      const listener = new ROSLIB.Topic({
        ros: this.ros,
        name: '/robot_to_webapp',
        messageType: 'std_msgs/String'
      });

      listener.subscribe((message) => {
        const parsedMessage = JSON.parse(message.data)
        if (parsedMessage.state === 'adaptability_response') {
          this.customOrder = parsedMessage.status;
        }
      });
      this.listener = listener;
    },
    cleartheinput() {
      this.transcript = '';
    },
    // Tap to dictate, tap again to stop: dictation stays open across pauses
    // instead of ending on the first one.
    toggleSpeechRecognition() {
      if (!this._dictation) return;
      if (!this._dictation.isActive() && this.$refs.textarea) {
        this.$refs.textarea.blur();
      }
      this._dictation.toggle();
    },
    initPublisher() {

      this.publisher = new ROSLIB.Topic({
        ros: this.ros,
        name: '/webapp_to_robot',
        messageType: 'std_msgs/String'
      })
    },
  }
}
</script>
