import { defineConfig } from 'vitepress'
import { getThemeConfig } from '@sugarat/theme/node'

// 博客主题独有配置（详见 https://theme.sugarat.top）
const blogTheme = getThemeConfig({
  // 关闭主题内置的 pagefind 搜索（其 Vue 组件在当前环境编译报错），
  // 沿用 themeConfig.search 配置的 VitePress 本地搜索
  search: false
})

export default defineConfig({
  // 继承博客主题；官方默认主题的 nav / sidebar / 中文文案依然生效
  extends: blogTheme,
  lang: 'zh-CN',
  title: '我的学习笔记',
  description: '记录学习过程',
  // 关闭「最后更新」时间：该时间取自 Git 提交记录，云端浅克隆会导致失真
  lastUpdated: false,
  vite: {
    ssr: {
      // 主题及其插件含 .vue 组件，交给 Vite 编译而非让 Node 直接加载，
      // 否则 SSR 阶段会报 Unknown file extension ".vue"
      noExternal: [/@sugarat\/theme/, /vitepress-plugin/]
    }
  },
  themeConfig: {
    search: { provider: 'local' },   // 开启全站本地搜索
    nav: [
      { text: '首页', link: '/' },
      { text: '笔记总览', link: '/notes/' },
      { text: '标签', link: '/tags' }
    ],
    // 只在 /notes/ 路径下显示笔记侧边栏，首页保持干净
    sidebar: {
      '/notes/': [
        {
          text: 'Python',
          collapsed: false,
          items: [
            { text: '01.智能聊天机器人体验', link: '/notes/python/01.智能聊天机器人体验' },
            { text: '02.python基础语法', link: '/notes/python/02.python基础语法' },
            { text: '03.python语句', link: '/notes/python/03.python语句' },
            { text: '04.for循环', link: '/notes/python/04.for循环' },
            { text: '05.容器与函数进阶', link: '/notes/python/05.容器与函数进阶' },
            { text: '06.函数与异常处理', link: '/notes/python/06.函数与异常处理' },
            { text: '07.面向对象', link: '/notes/python/07.面向对象' },
            { text: '08.闭包与装饰器', link: '/notes/python/08.闭包与装饰器' },
            { text: '09.网络编程', link: '/notes/python/09.网络编程' },
            { text: '10.多任务', link: '/notes/python/10.多任务' },
            { text: '11.协程与正则', link: '/notes/python/11.协程与正则' },
            { text: '12.pymysql与redis', link: '/notes/python/12.pymysql与redis' },
            { text: '13.数据分析三剑客', link: '/notes/python/13.数据分析三剑客' },
            { text: '14.linux指令', link: '/notes/python/14.linux指令' },
            { text: '15.linux基础', link: '/notes/python/15.linux基础' }
          ]
        },
        {
          text: 'MySQL',
          collapsed: false,
          items: [
            { text: '01.DDL+DML简单操作', link: '/notes/mysql/01.DDL+DML简单操作' },
            { text: '02.SQL约束', link: '/notes/mysql/02.SQL约束' },
            { text: '03.DQL单表查询', link: '/notes/mysql/03.DQL单表查询' },
            { text: '04.DQL多表查询', link: '/notes/mysql/04.DQL多表查询' },
            { text: '05.数据分析函数', link: '/notes/mysql/05.数据分析函数' }
          ]
        },
        {
          text: '机器学习 / 深度学习 / NLP',
          collapsed: false,
          items: [
            { text: '机器学习', link: '/notes/ML_DL_NLP/机器学习' },
            { text: '深度学习', link: '/notes/ML_DL_NLP/深度学习' },
            { text: 'NLP', link: '/notes/ML_DL_NLP/NLP' }
          ]
        },
        {
          text: 'RAG',
          collapsed: false,
          items: [
            { text: 'RAG', link: '/notes/RAG/RAG' }
          ]
        },
        {
          text: 'KnowForgeRag',
          collapsed: false,
          items: [
            { text: 'KnowForgeRag', link: '/notes/KnowForgeRag/KnowForgeRag' }
          ]
        },
        {
          text: '文本分类',
          collapsed: false,
          items: [
            { text: '文本分类', link: '/notes/text_classification_demo/文本分类' }
          ]
        }
      ]
    },
    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdatedText: '最后更新',
    darkModeSwitchLabel: '主题',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '回到顶部'
  }
})
