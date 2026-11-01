// Turns ```mermaid fenced code blocks into inline SVG at build time using Pele
// (https://pele.run). No client-side JS is shipped for diagrams.
import { visit } from 'unist-util-visit';
import { render } from 'pele';

export default function remarkPele() {
  return (tree, file) => {
    visit(tree, 'code', (node, index, parent) => {
      if (node.lang !== 'mermaid' || !parent || index === undefined) return;
      try {
        const { svg } = render(node.value);
        parent.children[index] = {
          type: 'html',
          value: `<figure class="pele-diagram not-content">${svg}</figure>`,
        };
      } catch (err) {
        // Keep the source visible instead of failing the whole build.
        file.message(`Pele could not render mermaid block: ${err}`, node);
      }
    });
  };
}
