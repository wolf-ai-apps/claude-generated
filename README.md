# Learning Docs

Astro + Starlight site. Mermaid blocks are rendered to SVG at build time with [Pele](https://pele.run).

```sh
npm install
npm run dev      # http://localhost:4321
npm run build    # static output in dist/
```

## Adding content

- Put `.md` / `.mdx` files in `src/content/docs/guides/` (or any subfolder).
- Each file needs frontmatter with at least a `title:`.
- To add a new sidebar group, create a folder and add it to `sidebar` in `astro.config.mjs`:
  `{ label: 'Kubernetes', autogenerate: { directory: 'kubernetes' } }`
