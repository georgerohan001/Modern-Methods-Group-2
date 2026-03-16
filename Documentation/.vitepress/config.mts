import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  // The 'base' must match your GitHub repository name exactly
  base: '/Modern-Methods-Group-2/', 
  
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
        text: '1. Introduction',
        items: [
          { text: '1.1 Project Motivation', link: '/motivation' },
        ]
      },
      {
        text: '2. The Pipeline',
        items: [
          { text: '2.2 Data Acquisition', link: '/data' },
          { text: '2.3 Preprocessing & Slicing', link: '/preprocessing' },
          { text: '2.4 Model Architecture', link: '/model' },
          { text: '2.5 Training Process', link: '/training' },
        ]
      },
      {
        text: '3. Outcomes',
        items: [
          { text: '3.1Results', link: '/results' },
          { text: '3.2 Discussion', link: '/discussion' },
          { text: '3.3 References', link: '/references' },
        ]
      }
    ],

    // Updated to reflect your actual repository link
    socialLinks: [
      { icon: 'github', link: 'https://github.com/georg/Modern-Methods-Group-2' }
    ],

    footer: {
      message: 'MSc Forestry: Modern Methods in TLS and UAV',
      copyright: 'Copyright © 2026 Group 1'
    }
  }
})