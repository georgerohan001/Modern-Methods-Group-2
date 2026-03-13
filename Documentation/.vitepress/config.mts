import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Modern Methods: Group 1",
  description: "Final paper: Forest surveying using TLS, UAVs, and YOLOv11 deep learning.",
  
  themeConfig: {
    // Top Navigation Bar
    nav: [
      { text: 'Home', link: '/' },
      { text: 'The Report', link: '/motivation' },
    ],

    // The Sidebar Menu
    sidebar: [
      {
        text: 'I. Introduction',
        items: [
          { text: 'Project Motivation', link: '/motivation' },
          { text: 'Data Acquisition', link: '/data' },
        ]
      },
      {
        text: 'II. The Pipeline',
        items: [
          { text: 'Preprocessing & Slicing', link: '/preprocessing' },
          { text: 'Model Architecture', link: '/model' },
          { text: 'Training Process', link: '/training' },
        ]
      },
      {
        text: 'III. Outcomes',
        items: [
          { text: 'Results', link: '/results' },
          { text: 'Discussion', link: '/discussion' },
        ]
      }
    ],

    // Update this link to your actual GitHub Repo later
    socialLinks: [
      { icon: 'github', link: 'https://github.com/georg/Modern-Methods-Group-1' }
    ],

    footer: {
      message: 'MSc Forestry: Modern Methods in TLS and UAV',
      copyright: 'Copyright © 2026 Group 1'
    }
  }
})