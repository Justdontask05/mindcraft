import OpenAIApi from 'openai';
import { getKey, hasKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js';

const DEFAULT_BASE_URL = 'https://api.minimax.io/v1';
const DEFAULT_MODEL = 'MiniMax-M2.7';

const normalizeModelName = (modelName) => {
    if (!modelName) {
        return DEFAULT_MODEL;
    }

    let normalized = modelName;

    if (normalized.startsWith('minimax/')) {
        normalized = normalized.slice('minimax/'.length);
    }

    if (normalized.toLowerCase().startsWith('minimax-')) {
        normalized = normalized.slice('minimax-'.length);
    }

    if (normalized.toLowerCase().startsWith('m2')) {
        normalized = `M${normalized.slice(1)}`;
    }

    if (!normalized.startsWith('MiniMax-')) {
        normalized = `MiniMax-${normalized}`;
    }

    return normalized;
};

export class Minimax {
    static prefix = 'minimax';

    constructor(model_name, url, params) {
        this.model_name = normalizeModelName(model_name);
        this.params = params;

        const apiKey = hasKey('MINIMAX_API_KEY')
            ? getKey('MINIMAX_API_KEY')
            : getKey('OPENAI_API_KEY');

        const config = {
            baseURL: url || DEFAULT_BASE_URL,
            apiKey,
            defaultHeaders: {
                Authorization: `Bearer ${apiKey}`
            }
        };

        this.openai = new OpenAIApi(config);
    }

    async sendRequest(turns, systemMessage, stop_seq = '***') {
        let messages = [{ role: 'system', content: systemMessage }, ...turns];
        messages = strictFormat(messages);

        const pack = {
            model: this.model_name || DEFAULT_MODEL,
            messages,
            stop: stop_seq,
            ...(this.params || {})
        };

        if (pack.temperature !== undefined && (pack.temperature <= 0 || pack.temperature > 1)) {
            console.warn('MiniMax temperature must be in (0, 1]. Clamping to 1.0.');
            pack.temperature = 1.0;
        }

        let res = null;
        try {
            console.log('Awaiting MiniMax api response...');
            const completion = await this.openai.chat.completions.create(pack);
            if (completion.choices[0].finish_reason === 'length') {
                throw new Error('Context length exceeded');
            }
            console.log('Received.');
            res = completion.choices[0].message.content;
            if (res.includes('</think>')) {
                if (!res.includes('<think>')) {
                    res = '<think>' + res;
                }
                res = res.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
            }
        } catch (err) {
            if ((err.message === 'Context length exceeded' || err.code === 'context_length_exceeded') && turns.length > 1) {
                console.log('Context length exceeded, trying again with shorter context.');
                return await this.sendRequest(turns.slice(1), systemMessage, stop_seq);
            }
            console.log(err);
            res = 'My brain disconnected, try again.';
        }

        return res;
    }

    async sendVisionRequest(_) {
        return 'Vision is only supported by certain models.';
    }

    async embed(text) {
        throw new Error('Embeddings are not supported by MiniMax.');
    }
}
