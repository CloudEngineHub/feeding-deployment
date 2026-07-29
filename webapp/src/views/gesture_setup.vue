<template>
  <div class="page">
    <div class="tb">
      <div class="av"><img src="../assets/user_avatar.svg" alt="User"></div>
      <div>
        <div class="tb-n">{{ username }}</div>
        <div class="tb-s">Describe the new gesture</div>
      </div>
    </div>

    <div class="bd">
      <div class="field-stack">
        <span class="field-lbl">Gesture name</span>
        <div class="field-box">
          <input
            type="text"
            v-model="transcript"
            placeholder="Typing..."
            ref="textarea1"
            class="field-input"
          >
          <button @click="toggleSpeechRecognition1"
                  class="icon-btn"
                  :class="{ 'amber-ic': isRecognizing1 }"
                  :title="isRecognizing1 ? 'stop voice' : 'voice'">
            <img alt="voice" src="../assets/voice.png">
          </button>
          <button @click="cleartheinput1" class="icon-btn" title="clear">
            <img alt="clear" src="../assets/clear.png">
          </button>
        </div>
        <p class="voice-status" :class="{ empty: !voiceStatus1 }" aria-live="polite">
          <img alt="" src="../assets/voice.png">
          <span v-if="voiceStatus1">{{ voiceStatus1 }}</span>
          <span v-else>&nbsp;</span>
        </p>

        <span class="field-lbl">Description</span>
        <div class="field-box tall">
          <textarea
            v-model="transcriptDes"
            placeholder="Typing..."
            ref="textarea2"
            class="field-input"
          ></textarea>
          <button @click="toggleSpeechRecognition2"
                  class="icon-btn"
                  :class="{ 'amber-ic': isRecognizing2 }"
                  :title="isRecognizing2 ? 'stop voice' : 'voice'">
            <img alt="voice" src="../assets/voice.png">
          </button>
          <button @click="cleartheinput2" class="icon-btn" title="clear">
            <img alt="clear" src="../assets/clear.png">
          </button>
        </div>
        <p class="voice-status" :class="{ empty: !voiceStatus2 }" aria-live="polite">
          <img alt="" src="../assets/voice.png">
          <span v-if="voiceStatus2">{{ voiceStatus2 }}</span>
          <span v-else>&nbsp;</span>
        </p>

        <div class="field-actions">
          <button class="btn md amber" style="width:35%" @click="confirmSelection">Next</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import ROSLIB from 'roslib'
import routeMap from '@/router/routeMap';
import { ROS_URL, USER} from '@/config/parameterConfig';
import { createDictation } from '@/utils/dictation';
export default {
  data () {
    return {
      ros: null,
      username: USER,
      isRecognizing1: false,
      isRecognizing2: false,
      listener: null,
      publisher: null,
      transcript: '',
      transcriptDes: '',
      voiceStatus1: '',
      voiceStatus2: '',
    }
  },
  mounted () {
    this.ros = new ROSLIB.Ros({ url: ROS_URL })
    this.initSubscriber()
    this.initPublisher()
    // Two independent fields, so two recognizers. Non-reactive on purpose:
    // they hold live SpeechRecognition objects.
    this._dictation1 = createDictation({
      onText: (text) => { this.transcript += text },
      onStatus: (status) => { this.voiceStatus1 = status },
      onActive: (active) => { this.isRecognizing1 = active }
    })
    this._dictation2 = createDictation({
      onText: (text) => { this.transcriptDes += text },
      onStatus: (status) => { this.voiceStatus2 = status },
      onActive: (active) => { this.isRecognizing2 = active }
    })
  },
  beforeUnmount () {
    if (this._dictation1) this._dictation1.destroy()
    if (this._dictation2) this._dictation2.destroy()
  },
  beforeRouteLeave (to, from, next) {
    if (this._dictation1) this._dictation1.destroy()
    if (this._dictation2) this._dictation2.destroy()

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
    cleartheinput1() {
      this.transcript = '';
    },

    cleartheinput2() {
      this.transcriptDes = '';
    },

    // Tap to dictate, tap again to stop: dictation stays open across pauses
    // instead of ending on the first one. Only one field at a time -- the two
    // recognizers would otherwise fight over the same microphone.
    toggleSpeechRecognition1() {
      if (!this._dictation1) return;
      if (!this._dictation1.isActive()) {
        if (this._dictation2) this._dictation2.stop();
        if (this.$refs.textarea1) this.$refs.textarea1.blur();
      }
      this._dictation1.toggle();
    },

    toggleSpeechRecognition2() {
      if (!this._dictation2) return;
      if (!this._dictation2.isActive()) {
        if (this._dictation1) this._dictation1.stop();
        if (this.$refs.textarea2) this.$refs.textarea2.blur();
      }
      this._dictation2.toggle();
    },
    handleRosMessage(message) {
      try {
        const parsedMessage = JSON.parse(message.data);
        const route = routeMap[parsedMessage.state]?.[parsedMessage.status];
        if (route) {
          if (typeof route === 'string') {
            this.$router.push(route);
          } else if (typeof route === 'object') {
            this.$router.push(route);
          }
        }
      } catch (error) {
      }
    },

    confirmSelection () {
      if (this.transcript !== '' && this.transcriptDes !== '') {
        const voiceMessage = new ROSLIB.Message({
          data: JSON.stringify({
            state: this.transcript,
            status: this.transcriptDes
          })
        });
        this.publisher.publish(voiceMessage);
      }
      this.transcript = '';
      this.transcriptDes = '';
      this.$router.push('/robot_executing');
    },
    initSubscriber() {

      this.listener = new ROSLIB.Topic({
        ros: this.ros,
        name: '/robot_to_webapp',
        messageType: 'std_msgs/String'
      })
      this.listener.subscribe((message) => {
        this.handleRosMessage(message);
      });
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

<style scoped>
.field-stack {
  max-width: 820px;
  width: 100%;
  margin: 0 auto;
}
</style>
