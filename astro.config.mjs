// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkPele from './src/plugins/remark-pele.mjs';

export default defineConfig({
  markdown: {
    // Runs before Starlight's Expressive Code, so ```mermaid blocks become SVG
    // instead of being highlighted as code.
    remarkPlugins: [remarkPele],
  },
  integrations: [
    starlight({
      title: 'Learning Docs',
      customCss: ['./src/styles/custom.css'],
      // Every .md/.mdx file under src/content/docs shows up automatically.
      // Folders become sidebar groups.
      sidebar: [
        { label: 'Guides', autogenerate: { directory: 'guides' } },
      ],
    }),
  ],
});
