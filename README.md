# EPUB Translator with Layout Preservation

This Python tool parses an EPUB eBook, translates its text using the OpenAI API, and repackages it into a new EPUB while preserving the original layout, structure, and formatting.  
It also supports context files for handling name changes, style guidance, and translation rules — ideal for book series where certain terms or names must be consistently adapted.

---

## ✨ Features
- **EPUB parsing and rebuilding** – preserves all original HTML, CSS, and metadata.
- **Chunk-based translation** – handles long texts by splitting them into manageable segments.
- **Poetry mode detection** – translates poems while keeping line breaks and rhythm.
- **Context file support** – provides the model with translation rules (e.g., name mappings, style notes).
- **Custom model choice** – defaults to `gpt-4o`, but any OpenAI model can be used.
- **Full logging** – see exactly which file and chunk is being translated.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/epub-translator.git
cd epub-translator

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or:
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

## 🛠 Usage

python aiepubtranslate.py \
  --input "Alice's Adventures in Wonderland.epub" \
  --output "Alice im Wunderland.epub" \
  --target-lang "de" \
  --source-lang "en" \
  --model "gpt-4o" \
  --api-key "sk-YOUR_OPENAI_KEY" \
  --context-file "context.txt"

Arguments:

--input – Path to the source EPUB file.
--output – Path where the translated EPUB will be saved.
--target-lang – Target language code (e.g., de, fr, es).
--source-lang – Source language code (optional; auto if omitted).
--model – OpenAI model name (e.g., gpt-4o, gpt-4.1-mini).
--api-key – Your OpenAI API key (passed directly, not from env vars).
--context-file – Optional .txt file containing context instructions and name mappings.
```

## 📄 Example context file:
```
The translation target language is German.
Maintain the style of a children's fairy tale, using natural, fluid German.
Keep sentence structure close to the original where possible, but prioritize readability over strict word-for-word translation.
Preserve paragraph breaks, punctuation, and all formatting exactly as in the source.
Do not translate names unless specified below. Use the exact German equivalents for the following recurring characters:

Alice → Alice
Cheshire Cat → Grinsekatze
White Rabbit → Weißes Kaninchen
Mad Hatter → Verrückter Hutmacher
March Hare → Faselhase
Queen of Hearts → Herzkönigin
King of Hearts → Herzkönig
Duchess → Herzogin
Cook → Köchin
Mock Turtle → Schein-Schildkröte
Gryphon → Greif
Caterpillar → Raupe
Bill the Lizard → Bill die Eidechse
Dinah → Dina
Knave of Hearts → Herzbube

For poems, riddles, or songs, translate them as German poems suitable for children, preserving rhyme and rhythm where possible, but prioritize creative equivalence over literal accuracy.
Do not add explanations, translator's notes, or any additional commentary.
```

## 🔍 Logging
The script outputs detailed logs:
- Which file is currently being processed
- Chunk sizes and poetry detection
- Any translation length mismatches and reconciliation attempts
- Preview of raw model output when problems occur

## ⚠️ Notes
- OpenAI API usage costs depend on the book length, chosen model, and token count.
- As a rough guide, a short children’s book (~10k tokens) with gpt-4o might cost less than $0.50 to translate.
- The tool will retry failed translations up to 3 times before raising an error.
- Only the text nodes are translated; images, CSS, and formatting are preserved.
