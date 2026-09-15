with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

cover_block = (
    "\n<div align=\"center\">\n\n"
    "![SentinelEdge — Bimanual VLA on Intel Core Ultra](assets/cover.jpg)\n\n"
    "</div>\n\n"
)

old = "## Bimanual VLA Manipulation with Multi-Modal Reasoning\n"
new = "## Bimanual VLA Manipulation with Multi-Modal Reasoning\n" + cover_block

content = content.replace(old, new, 1)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Cover image injected into README successfully.")
