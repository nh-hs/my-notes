import { createContentLoader } from 'vitepress'

export interface NoteItem {
  title: string
  url: string
  tags: string[]
  category: string
}

declare const data: NoteItem[]
export { data }

// 分类目录 -> 中文显示名
const CATEGORY_LABEL: Record<string, string> = {
  python: 'Python',
  mysql: 'MySQL',
  ML_DL_NLP: '机器学习 / 深度学习 / NLP',
  RAG: 'RAG',
  KnowForgeRag: 'KnowForgeRag',
  text_classification_demo: '文本分类'
}

export default createContentLoader('notes/**/*.md', {
  transform(raw): NoteItem[] {
    return raw
      .filter((page) => !!page.frontmatter.title)
      .map((page) => {
        // url 形如 /notes/python/02.python基础语法.html
        const dir = page.url.split('/')[2] || ''
        return {
          title: page.frontmatter.title,
          url: page.url,
          tags: page.frontmatter.tags || [],
          category: CATEGORY_LABEL[dir] || dir
        }
      })
      .sort((a, b) => a.category.localeCompare(b.category, 'zh') || a.title.localeCompare(b.title, 'zh'))
  }
})
