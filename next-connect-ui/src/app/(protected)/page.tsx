'use client'

import Link from 'next/link'
import { FileText, Folder, Search, FlaskConical, Github, ExternalLink } from 'lucide-react'
import { useTranslation } from '@/hooks/use-translation'

export default function MainPage() {
  const { t } = useTranslation()

  return (
    <div className="min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            {t('main.title')}
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            <span dangerouslySetInnerHTML={{ __html: t('main.subtitle') }} />
            <br />
            {t('main.description')}
          </p>
        </div>

        <div className="mb-12">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mb-6">{t('main.keyFeatures')}</h2>
          <p className="text-gray-600 dark:text-gray-300 mb-8">
            {t('main.keyFeaturesDescription')}
          </p>

          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white dark:bg-card rounded-lg shadow-md dark:shadow-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <Folder className="w-6 h-6 text-blue-600" />
                <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100">{t('main.collectionManagement.title')}</h3>
              </div>
              <ul className="text-gray-600 dark:text-gray-300 space-y-2 mb-4">
                {(t('main.collectionManagement.features') as unknown as string[]).map((feature, index) => (
                  <li key={index}>• {feature}</li>
                ))}
              </ul>
              <Link
                href="/collections"
                className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-700 font-medium"
              >
                {t('main.collectionManagement.goTo')}
                <Folder className="w-4 h-4" />
              </Link>
            </div>

            <div className="bg-white dark:bg-card rounded-lg shadow-md dark:shadow-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <FileText className="w-6 h-6 text-green-600" />
                <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100">{t('main.documentManagement.title')}</h3>
              </div>
              <ul className="text-gray-600 dark:text-gray-300 space-y-2 mb-4">
                {(t('main.documentManagement.features') as unknown as string[]).map((feature, index) => (
                  <li key={index}>• {feature}</li>
                ))}
              </ul>
              <Link
                href="/documents"
                className="inline-flex items-center gap-2 text-green-600 hover:text-green-700 font-medium"
              >
                {t('main.documentManagement.goTo')}
                <FileText className="w-4 h-4" />
              </Link>
            </div>

            <div className="bg-white dark:bg-card rounded-lg shadow-md dark:shadow-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <Search className="w-6 h-6 text-purple-600" />
                <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100">{t('main.search.title')}</h3>
              </div>
              <ul className="text-gray-600 dark:text-gray-300 space-y-2 mb-4">
                {(t('main.search.features') as unknown as string[]).map((feature, index) => (
                  <li key={index} dangerouslySetInnerHTML={{ __html: `• ${feature}` }} />
                ))}
              </ul>
              <Link
                href="/search"
                className="inline-flex items-center gap-2 text-purple-600 hover:text-purple-700 font-medium"
              >
                {t('main.search.goTo')}
                <Search className="w-4 h-4" />
              </Link>
            </div>

            <div className="bg-white dark:bg-card rounded-lg shadow-md dark:shadow-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <FlaskConical className="w-6 h-6 text-orange-600" />
                <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100">{t('main.apiTester.title')}</h3>
              </div>
              <ul className="text-gray-600 dark:text-gray-300 space-y-2 mb-4">
                {(t('main.apiTester.features') as unknown as string[]).map((feature, index) => (
                  <li key={index}>• {feature}</li>
                ))}
              </ul>
              <Link
                href="/api-tester"
                className="inline-flex items-center gap-2 text-orange-600 hover:text-orange-700 font-medium"
              >
                {t('main.apiTester.goTo')}
                <FlaskConical className="w-4 h-4" />
              </Link>
            </div>
          </div>
        </div>

        <div className="border-t border-gray-200 pt-12">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mb-6">{t('main.about.title')}</h2>
          
          <div className="grid md:grid-cols-[2fr_1fr] gap-8">
            <div>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                <span dangerouslySetInnerHTML={{ __html: t('main.about.description') }} />
              </p>
              <ul className="space-y-3 text-gray-600 dark:text-gray-300">
                {(t('main.about.techStack') as unknown as string[]).map((tech, index) => {
                  const emoji = ['🦜', '🐘', '⚡', '🎨', '🎨'][index]
                  return (
                    <li key={index} className="flex items-start gap-2">
                      <span className="text-xl">{emoji}</span>
                      <span dangerouslySetInnerHTML={{ __html: tech }} />
                    </li>
                  )
                })}
              </ul>
              <p className="text-gray-600 dark:text-gray-300 mt-4">
                {t('main.about.ragReady')}
              </p>
            </div>

            <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">{t('main.about.links.title')}</h3>
              <ul className="space-y-3">
                <li>
                  <a
                    href="https://github.com/teddynote-lab/LangConnect-Client"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    <Github className="w-4 h-4" />
                    {t('main.about.links.github')}
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-16 text-center text-gray-500 dark:text-gray-400 text-sm">
          {t('main.footer')}{' '}
          <a
            href="https://github.com/teddynote-lab"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
          >
            TeddyNote LAB
          </a>
        </div>
      </div>
    </div>
  )
}