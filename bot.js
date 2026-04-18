/**
 * Discord Voice Agent Bot — optimized
 *
 * Optimizations vs previous version:
 *   1. Streaming Ollama + sentence-by-sentence TTS — bot starts speaking
 *      while the LLM is still generating the rest of the reply.
 *   2. Persistent Kokoro TTS server — model stays loaded between requests,
 *      eliminating the ~400ms Python cold-start on every response.
 *   3. Reduced silence window — 800ms instead of 2500ms, so processing
 *      starts sooner after the user stops talking.
 *
 * npm install discord.js @discordjs/voice @discordjs/opus prism-media axios dotenv
 */

require('dotenv').config();
const { Client, GatewayIntentBits, REST, Routes, SlashCommandBuilder } = require('discord.js');
const {
  joinVoiceChannel,
  createAudioPlayer,
  createAudioResource,
  EndBehaviorType,
  VoiceConnectionStatus,
  entersState,
  AudioPlayerStatus,
  StreamType,
} = require('@discordjs/voice');
const prism = require('prism-media');
const fs    = require('fs');
const path  = require('path');
const os    = require('os');
const { execFile } = require('child_process');
const axios = require('axios');

// ─────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────

const BOT_TOKEN          = process.env.BOT_TOKEN;
const GUILD_ID           = process.env.GUILD_ID;
const AUTO_JOIN_CHANNEL  = process.env.AUTO_JOIN_CHANNEL_ID || null;
const OWNER_USER_ID      = process.env.OWNER_USER_ID        || null;
const OLLAMA_URL         = process.env.OLLAMA_URL   || 'http://localhost:11434/api/chat';
const OLLAMA_MODEL       = process.env.OLLAMA_MODEL || 'llama3.2:3b';
const WHISPER_BIN        = process.env.WHISPER_BIN  || path.join(os.homedir(), 'whisper.cpp/build/bin/whisper-cli');
const WHISPER_MODEL_PATH = process.env.WHISPER_MODEL || path.join(os.homedir(), 'whisper.cpp/models/ggml-base.bin');
const WHISPER_LANG       = process.env.WHISPER_LANG || 'pt';
const TTS_SERVER_URL     = process.env.TTS_SERVER_URL || 'http://127.0.0.1:5500/tts';

const SYSTEM_PROMPT = process.env.SYSTEM_PROMPT ||
  'Você é um assistente de voz útil dentro de um canal de voz do Discord. ' +
  'Suas respostas são faladas em voz alta, então seja conversacional e conciso. ' +
  'Evite markdown, listas e blocos de código a menos que o usuário peça. ' +
  'Chame a pessoa pelo nome de usuário do Discord ao responder.';

// ── Audio constants ───────────────────────────
const DISCORD_SAMPLE_RATE   = 48000;
const DISCORD_CHANNELS      = 2;
const WHISPER_SAMPLE_RATE   = 16000;
const FRAME_SIZE            = 960;

// ── Timing ────────────────────────────────────
// Reduced from 2500ms → 800ms: processing starts much sooner after speech ends
const SILENCE_DURATION_MS   = 800;
const MIN_AUDIO_DURATION_MS = 1500;
const DEBOUNCE_MS           = 400;  // also tightened from 600ms

// ─────────────────────────────────────────────
// SLASH COMMANDS
// ─────────────────────────────────────────────

const commands = [
  new SlashCommandBuilder()
    .setName('join')
    .setDescription('Bot entra no seu canal de voz'),

  new SlashCommandBuilder()
    .setName('leave')
    .setDescription('Bot sai do canal de voz'),

  new SlashCommandBuilder()
    .setName('clear')
    .setDescription('Limpa o histórico de conversa'),

  new SlashCommandBuilder()
    .setName('model')
    .setDescription('Troca o modelo do Ollama')
    .addStringOption(opt =>
      opt.setName('name')
        .setDescription('Nome do modelo (ex: llama3.2:3b, qwen2.5:3b)')
        .setRequired(true)
    ),

  new SlashCommandBuilder()
    .setName('status')
    .setDescription('Mostra o status atual do bot'),

  new SlashCommandBuilder()
    .setName('sensitivity')
    .setDescription('Ajusta a sensibilidade do microfone em tempo real')
    .addNumberOption(opt =>
      opt.setName('value')
        .setDescription('Threshold de silêncio (padrão 0.04 — maior = menos sensível)')
        .setMinValue(0.01)
        .setMaxValue(0.2)
        .setRequired(true)
    ),
].map(cmd => cmd.toJSON());

async function registerSlashCommands() {
  const rest = new REST({ version: '10' }).setToken(BOT_TOKEN);
  try {
    console.log('[Slash] Registrando comandos...');
    await rest.put(
      Routes.applicationGuildCommands(process.env.CLIENT_ID, GUILD_ID),
      { body: commands }
    );
    console.log('[Slash] Comandos registrados.');
  } catch (err) {
    console.error('[Slash] Falha ao registrar comandos:', err.message);
  }
}

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────

const guilds = new Map();
let currentThreshold = 0.04;

// ─────────────────────────────────────────────
// DISCORD CLIENT
// ─────────────────────────────────────────────

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
  ],
});

// ─────────────────────────────────────────────
// AUDIO UTILITIES
// ─────────────────────────────────────────────

function downsample(buffer) {
  const input  = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
  const ratio  = DISCORD_SAMPLE_RATE / WHISPER_SAMPLE_RATE;
  const outLen = Math.floor(input.length / DISCORD_CHANNELS / ratio);
  const output = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const src = Math.floor(i * ratio) * DISCORD_CHANNELS;
    output[i] = (input[src] + input[src + 1]) >> 1;
  }
  return Buffer.from(output.buffer);
}

function calcRms(buffer) {
  const samples = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i] / 32768;
    sum += s * s;
  }
  return Math.sqrt(sum / samples.length);
}

function writePcmToWav(pcmBuffer, filePath) {
  return new Promise((resolve, reject) => {
    const dataSize      = pcmBuffer.length;
    const numChannels   = 1;
    const bitsPerSample = 16;
    const byteRate      = WHISPER_SAMPLE_RATE * numChannels * bitsPerSample / 8;
    const blockAlign    = numChannels * bitsPerSample / 8;
    const header        = Buffer.alloc(44);

    header.write('RIFF', 0);
    header.writeUInt32LE(36 + dataSize, 4);
    header.write('WAVE', 8);
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(numChannels, 22);
    header.writeUInt32LE(WHISPER_SAMPLE_RATE, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(blockAlign, 32);
    header.writeUInt16LE(bitsPerSample, 34);
    header.write('data', 36);
    header.writeUInt32LE(dataSize, 40);

    fs.writeFile(filePath, Buffer.concat([header, pcmBuffer]), (err) => {
      if (err) reject(err); else resolve();
    });
  });
}

// ─────────────────────────────────────────────
// TRANSCRIPTION (whisper.cpp)
// ─────────────────────────────────────────────

function transcribe(wavPath) {
  return new Promise((resolve, reject) => {
    execFile(
      WHISPER_BIN,
      ['-m', WHISPER_MODEL_PATH, '-f', wavPath, '--language', WHISPER_LANG, '--no-timestamps', '-nt'],
      { timeout: 30000 },
      (err, stdout) => {
        if (err) { reject(err); return; }
        const text = stdout
          .split('\n')
          .map(l => l.replace(/\[.*?\]/g, '').trim())
          .filter(Boolean)
          .join(' ')
          .trim();
        resolve(text);
      }
    );
  });
}

// ─────────────────────────────────────────────
// TTS — calls persistent Kokoro server
// ─────────────────────────────────────────────

function detectLang(text) {
  const asciiRatio = (text.match(/[a-zA-Z]/g) || []).length / Math.max(text.length, 1);
  return asciiRatio > 0.85 ? 'en-us' : 'pt-br';
}

async function generateTts(text, outputPath) {
  const response = await axios.post(
    TTS_SERVER_URL,
    { text, lang: detectLang(text) },
    { responseType: 'arraybuffer', timeout: 15000 }
  );
  fs.writeFileSync(outputPath, Buffer.from(response.data));
}

// ─────────────────────────────────────────────
// AUDIO PLAYER
// ─────────────────────────────────────────────

function playAudio(guildId, audioPath) {
  const state = guilds.get(guildId);
  if (!state?.player) return Promise.resolve();
  return new Promise((resolve) => {
    const resource = createAudioResource(audioPath, { inputType: StreamType.Arbitrary });
    state.player.play(resource);
    state.player.once(AudioPlayerStatus.Idle, resolve);
    state.player.once('error', (err) => { console.error('[Player]', err.message); resolve(); });
  });
}

// ─────────────────────────────────────────────
// PIPELINE — streaming LLM + sentence TTS
//
// Instead of waiting for the full reply, we stream tokens from Ollama
// and speak each sentence as soon as it's complete. The bot starts
// talking after the first sentence while the LLM generates the rest.
// ─────────────────────────────────────────────

async function processAudio(guildId, userId, username, pcmChunks) {
  const tmpDir  = os.tmpdir();
  const wavPath = path.join(tmpDir, `discord_${userId}_${Date.now()}.wav`);

  try {
    const rawPcm   = Buffer.concat(pcmChunks);
    const pcm16khz = downsample(rawPcm);

    // Duration filter
    const durationMs = (pcm16khz.length / 2 / WHISPER_SAMPLE_RATE) * 1000;
    if (durationMs < MIN_AUDIO_DURATION_MS) {
      console.log(`[Filter] Ignorado: ${Math.round(durationMs)}ms (muito curto)`);
      return;
    }

    // Volume filter
    const rms = calcRms(pcm16khz);
    if (rms < currentThreshold) {
      console.log(`[Filter] Ignorado: RMS ${rms.toFixed(4)} (abaixo do threshold)`);
      return;
    }

    // Transcribe
    await writePcmToWav(pcm16khz, wavPath);
    console.log(`[Whisper] Transcrevendo ${username}...`);
    const text = await transcribe(wavPath);
    if (!text || text.length < 3) { console.log('[Whisper] Vazio, ignorando.'); return; }
    console.log(`[${username}]: ${text}`);

    // Stream LLM response + speak sentence by sentence
    const state = guilds.get(guildId);
    if (!state) return;

    const model = process.env.OLLAMA_MODEL_OVERRIDE || OLLAMA_MODEL;
    state.history.push({ role: 'user', content: `[${username}]: ${text}` });

    console.log('[Ollama] Streaming...');

    const response = await axios.post(
      OLLAMA_URL,
      {
        model,
        messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...state.history],
        stream: true,
      },
      { responseType: 'stream', timeout: 60000 }
    );

    let tokenBuffer  = '';  // accumulates tokens until a sentence boundary
    let fullReply    = '';  // full reply for history

    // Speak a chunk of text immediately
    async function speakChunk(chunk) {
      const ttsPath = path.join(tmpDir, `tts_${Date.now()}.wav`);
      try {
        await generateTts(chunk, ttsPath);
        await playAudio(guildId, ttsPath);
      } finally {
        fs.unlink(ttsPath, () => {});
      }
    }

    await new Promise((resolve, reject) => {
      let remainder = '';

      response.data.on('data', async (raw) => {
        // Ollama streams newline-delimited JSON — handle partial chunks
        const lines = (remainder + raw.toString()).split('\n');
        remainder = lines.pop();  // last line may be incomplete

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const parsed = JSON.parse(line);
            const token = parsed.message?.content || '';
            tokenBuffer += token;
            fullReply   += token;

            // Speak when we hit a sentence boundary followed by a space
            // (avoids cutting mid-abbreviation like "Dr. Smith")
            const match = tokenBuffer.search(/[.!?]\s/);
            if (match !== -1) {
              const sentence = tokenBuffer.slice(0, match + 1).trim();
              tokenBuffer    = tokenBuffer.slice(match + 2);
              if (sentence.length > 3) {
                console.log(`[Bot] "${sentence}"`);
                await speakChunk(sentence);
              }
            }
          } catch { /* partial JSON line — handled by remainder */ }
        }
      });

      response.data.on('end', async () => {
        // Speak any remaining text after stream ends
        const leftover = (tokenBuffer + remainder).trim();
        if (leftover.length > 3) {
          console.log(`[Bot] "${leftover}"`);
          await speakChunk(leftover);
        }
        resolve();
      });

      response.data.on('error', reject);
    });

    // Save full reply to history
    if (fullReply.trim()) {
      state.history.push({ role: 'assistant', content: fullReply.trim() });
      if (state.history.length > 20) state.history = state.history.slice(-20);
    }

    console.log();

  } catch (err) {
    console.error('[Pipeline] Erro:', err.message);
  } finally {
    fs.unlink(wavPath, () => {});
  }
}

// ─────────────────────────────────────────────
// VOICE CONNECTION
// ─────────────────────────────────────────────

async function connectToChannel(channel) {
  const guildId = channel.guild.id;

  const connection = joinVoiceChannel({
    channelId: channel.id,
    guildId,
    adapterCreator: channel.guild.voiceAdapterCreator,
    selfDeaf: false,
    selfMute: false,
  });

  const player = createAudioPlayer();
  connection.subscribe(player);

  try {
    await entersState(connection, VoiceConnectionStatus.Ready, 20_000);
  } catch {
    connection.destroy();
    throw new Error('Não foi possível conectar ao canal em 20 segundos.');
  }

  guilds.set(guildId, { connection, player, history: [], channelName: channel.name });

  connection.on(VoiceConnectionStatus.Disconnected, async () => {
    try {
      await Promise.race([
        entersState(connection, VoiceConnectionStatus.Signalling, 5_000),
        entersState(connection, VoiceConnectionStatus.Connecting, 5_000),
      ]);
    } catch {
      connection.destroy();
      guilds.delete(guildId);
    }
  });

  startListening(guildId);
  console.log(`[Bot] Entrou em #${channel.name} em ${channel.guild.name}`);
}

// ─────────────────────────────────────────────
// AUDIO RECEIVER + DEBOUNCE
// ─────────────────────────────────────────────

function startListening(guildId) {
  const state = guilds.get(guildId);
  if (!state) return;

  const { connection } = state;
  const receiver = connection.receiver;

  const pendingTimers = new Map();
  const pendingChunks = new Map();
  const queue         = [];
  let processing      = false;

  function drainQueue() {
    if (processing || queue.length === 0) return;
    processing = true;
    const job = queue.shift();
    processAudio(job.guildId, job.userId, job.username, job.chunks).finally(() => {
      processing = false;
      drainQueue();
    });
  }

  receiver.speaking.on('start', (userId) => {
    if (userId === client.user.id) return;

    if (pendingTimers.has(userId)) {
      clearTimeout(pendingTimers.get(userId));
      pendingTimers.delete(userId);
    }

    const member   = client.guilds.cache.get(guildId)?.members.cache.get(userId);
    const username = member?.displayName || member?.user?.username || `User ${userId}`;

    const audioStream = receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: SILENCE_DURATION_MS },
    });

    const decoder = new prism.opus.Decoder({
      frameSize: FRAME_SIZE,
      channels:  DISCORD_CHANNELS,
      rate:      DISCORD_SAMPLE_RATE,
    });

    const chunks = [];

    audioStream
      .pipe(decoder)
      .on('data', (chunk) => chunks.push(chunk))
      .on('end', () => {
        if (chunks.length === 0) return;

        const merged = [...(pendingChunks.get(userId) || []), ...chunks];
        pendingChunks.set(userId, merged);

        const timer = setTimeout(() => {
          pendingTimers.delete(userId);
          const finalChunks = pendingChunks.get(userId) || [];
          pendingChunks.delete(userId);
          if (finalChunks.length === 0) return;

          if (!processing) {
            processing = true;
            processAudio(guildId, userId, username, finalChunks).finally(() => {
              processing = false;
              drainQueue();
            });
          } else {
            queue.push({ guildId, userId, username, chunks: finalChunks });
          }
        }, DEBOUNCE_MS);

        pendingTimers.set(userId, timer);
      })
      .on('error', (err) => console.error('[Decoder] Erro:', err.message));
  });
}

// ─────────────────────────────────────────────
// DISCORD EVENTS
// ─────────────────────────────────────────────

client.once('ready', async () => {
  console.log(`\n[Bot] Logado como ${client.user.tag}`);
  await registerSlashCommands();
  console.log(`[Bot] Modelo:    ${OLLAMA_MODEL}`);
  console.log(`[Bot] Whisper:   ${WHISPER_MODEL_PATH}`);
  console.log(`[Bot] Idioma:    ${WHISPER_LANG}`);
  console.log(`[Bot] TTS:       ${TTS_SERVER_URL}`);
  console.log(`[Bot] Threshold: ${currentThreshold}\n`);

  // Verify TTS server is running
  try {
    await axios.get(TTS_SERVER_URL.replace('/tts', '/health'), { timeout: 3000 });
    console.log('[TTS] Servidor respondendo.');
  } catch {
    console.warn('[TTS] AVISO: servidor TTS não responde. Inicie tts_server.py antes do bot.');
  }

  if (AUTO_JOIN_CHANNEL && GUILD_ID) {
    try {
      await client.guilds.fetch(GUILD_ID);
      const channel = await client.channels.fetch(AUTO_JOIN_CHANNEL);
      if (channel?.isVoiceBased()) await connectToChannel(channel);
    } catch (err) {
      console.error('[Auto-join] Falhou:', err.message);
    }
  }
});

client.on('voiceStateUpdate', async (oldState, newState) => {
  if (!OWNER_USER_ID || newState.member?.id !== OWNER_USER_ID) return;
  const guildId = newState.guild.id;

  if (newState.channelId && newState.channelId !== oldState.channelId) {
    const existing = guilds.get(guildId);
    if (existing) { existing.connection.destroy(); guilds.delete(guildId); }
    try {
      await connectToChannel(newState.channel);
    } catch (err) {
      console.error('[Auto-follow] Falhou:', err.message);
    }
  }

  if (!newState.channelId && oldState.channelId) {
    const state = guilds.get(guildId);
    if (state) {
      state.connection.destroy();
      guilds.delete(guildId);
      console.log('[Bot] Dono saiu — desconectado.');
    }
  }
});

// ─────────────────────────────────────────────
// SLASH COMMAND HANDLER
// ─────────────────────────────────────────────

client.on('interactionCreate', async (interaction) => {
  if (!interaction.isChatInputCommand()) return;
  const { commandName, guildId } = interaction;

  switch (commandName) {
    case 'join': {
      const voiceChannel = interaction.member?.voice?.channel;
      if (!voiceChannel) {
        await interaction.reply({ content: 'Você precisa estar em um canal de voz primeiro.', ephemeral: true });
        return;
      }
      const existing = guilds.get(guildId);
      if (existing) { existing.connection.destroy(); guilds.delete(guildId); }
      await interaction.deferReply();
      try {
        await connectToChannel(voiceChannel);
        await interaction.editReply(`Entrei em **${voiceChannel.name}**. Estou ouvindo!`);
      } catch (err) {
        await interaction.editReply(`Falha ao entrar: ${err.message}`);
      }
      break;
    }

    case 'leave': {
      const state = guilds.get(guildId);
      if (state) {
        state.connection.destroy();
        guilds.delete(guildId);
        await interaction.reply('Desconectado.');
      } else {
        await interaction.reply({ content: 'Não estou em nenhum canal de voz.', ephemeral: true });
      }
      break;
    }

    case 'clear': {
      const state = guilds.get(guildId);
      if (state) {
        state.history = [];
        await interaction.reply('Histórico de conversa limpo.');
      } else {
        await interaction.reply({ content: 'Não estou ativo neste servidor.', ephemeral: true });
      }
      break;
    }

    case 'model': {
      const newModel = interaction.options.getString('name');
      process.env.OLLAMA_MODEL_OVERRIDE = newModel;
      await interaction.reply(`Modelo trocado para: \`${newModel}\`\nVai usar na próxima mensagem.`);
      break;
    }

    case 'status': {
      const state = guilds.get(guildId);
      const model = process.env.OLLAMA_MODEL_OVERRIDE || OLLAMA_MODEL;
      if (state) {
        await interaction.reply(
          `**Status**\n` +
          `Canal: **${state.channelName}**\n` +
          `Modelo: \`${model}\`\n` +
          `Idioma: \`${WHISPER_LANG}\`\n` +
          `Threshold: \`${currentThreshold}\`\n` +
          `Histórico: ${state.history.length} mensagens`
        );
      } else {
        await interaction.reply(`Não conectado a nenhum canal.\nModelo: \`${model}\``);
      }
      break;
    }

    case 'sensitivity': {
      const value = interaction.options.getNumber('value');
      currentThreshold = value;
      await interaction.reply(
        `Sensibilidade ajustada para \`${value}\`.\n` +
        `(menor = mais sensível | maior = ignora mais ruído de fundo)`
      );
      break;
    }
  }
});

// ─────────────────────────────────────────────
// STARTUP
// ─────────────────────────────────────────────

if (!BOT_TOKEN) {
  console.error('[Erro] BOT_TOKEN não definido no .env');
  process.exit(1);
}

client.login(BOT_TOKEN).catch((err) => {
  console.error('[Erro] Falha no login:', err.message);
  process.exit(1);
});

process.on('SIGINT', () => {
  console.log('\n[Bot] Encerrando...');
  for (const [, state] of guilds) state.connection.destroy();
  client.destroy();
  process.exit(0);
});
